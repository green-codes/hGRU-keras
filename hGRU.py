"""
source: arXiv:1805.08315v4
additional reference: https://github.com/josueortc/pathfinder (pytorch implementation of hGRU)
## Notes
- channel-symmetric weights cannot be properly implemented without low-level modifications to the 
  computation graph
    - in tensorflow this is done by getting the graph with tf.get_default_graph() (depracated), 
      then modify the graph using Graph.gradient_override_map({'Conv2D': 'ChannelSymmetricConv'})
        - also requires disabling eager execution in TF2.0
        - see gamma-netlayers/recurrent/hgru-bn-for.py for source
    - in one pytorch implementation (serre-lab/hgru-pytorch) this is done via register_hook()
        - not the 'proper' implementation, tying the gradients together by averaging
    - implementing channel symmetry constraint by averaging between-channel weights on either 
      direction at each timestep
"""

import random
import numpy as np

# explicitly import tensorflow.keras to fix compatibility issues
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution() # a lot faster using static graphs
print("hGRU using Keras backend:", keras.backend.__name__)

@tf.custom_gradient
def channel_sym_conv2d(x, w): 
    with tf.GradientTape() as gx, tf.GradientTape() as gw:
        gx.watch(x)
        gw.watch(w)
        y = tf.nn.conv2d(x, w, 1, 'SAME')
    def grad(dy):
        dx = gx.gradient(y,x)
        dw = gw.gradient(y,w)
        dw = (dw + tf.transpose(dw, perm=[0,1,3,2])) * 0.5 # tie gradients
        return dx, dw
    return tf.identity(y), grad

class hGRUCell(keras.layers.Layer):

    def __init__(self, spatial_extent=5, timesteps=8, batchnorm=False, 
                 channel_sym=True, rand_seed=None, **kwargs):
        
        self.spatial_extent = spatial_extent
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.channel_sym = channel_sym
        self.rand_seed = rand_seed if rand_seed else np.uintc(hash(random.random()))
        
        super(hGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # NOTE: assume channel-last inputs
        # NOTE: make sure that all initializer seeds are different! 
        
        # U(1) and U(2): 1x1xKxK kernels; b(1) and b(2): 1x1xK channel-wise gate biases
        # NOTE: use chronos initialization for biases
        self.u1 = self.add_weight(name='u1', 
                                  shape=(1,1,input_shape[-1], input_shape[-1]),
                                  initializer=keras.initializers.Orthogonal(seed=self.rand_seed),
                                  trainable=True)
        self.u2 = self.add_weight(name='u2', 
                                  shape=(1,1,input_shape[-1], input_shape[-1]),
                                  initializer=keras.initializers.Orthogonal(seed=self.rand_seed+1),
                                  trainable=True)
        self.b1 = self.add_weight(name='b1', 
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        self.b2 = self.add_weight(name='b2', 
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        self.b1.assign(K.log(keras.initializers.RandomNormal(1,self.timesteps-1,
                                                             seed=self.rand_seed+5)(self.b1.shape)))
        self.b2.assign(-self.b1)


        # one separate batchnorm layer for each timestep
        self.bn = [keras.layers.BatchNormalization(momentum=0.001, epsilon=1e-03) 
                   for _ in range(self.timesteps*4)]
        
        # W: SxSxKxK shared inhibition/excitation kernel
        self.w_inh = self.add_weight(name='w_inh', 
                      shape=(self.spatial_extent, self.spatial_extent, 
                             input_shape[-1], input_shape[-1]),
                      initializer=keras.initializers.Orthogonal(seed=self.rand_seed+2),
                      trainable=True)
        self.w_exc = self.add_weight(name='w_exc', 
                      shape=(self.spatial_extent, self.spatial_extent, 
                             input_shape[-1], input_shape[-1]),
                      initializer=keras.initializers.Orthogonal(seed=self.rand_seed+3),
                      trainable=True)
        # symmetric init 
        if self.channel_sym:
            self.w_inh.assign((self.w_inh + K.permute_dimensions(self.w_inh, (0,1,3,2))) * 0.5)
            self.w_exc.assign((self.w_exc + K.permute_dimensions(self.w_exc, (0,1,3,2))) * 0.5)
        
        # mu, alpha: channel-wise linear/quadratic control for inhibition
        self.mu = self.add_weight(name='mu',
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        self.alpha = self.add_weight(name='alpha',
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        
        # kappa, omega, beta: channel-wise linear/quadratic control and additional gain for excitation
        self.kappa = self.add_weight(name='kappa',
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        self.omega = self.add_weight(name='omega',
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)
        self.beta = self.add_weight(name='beta',    # TODO: initialize beta as ones?
                                  shape=(1,1,input_shape[-1]),
                                  trainable=True)

        self.mu.assign(K.ones(self.mu.shape) * 1.0)
        self.alpha.assign(K.ones(self.alpha.shape) * 0.1)
        self.kappa.assign(K.ones(self.kappa.shape) * 0.5)
        self.omega.assign(K.ones(self.omega.shape) * 0.5)
        self.beta.assign(K.ones(self.beta.shape) * 1.0)
        
        # eta: timestep weights
        if not self.batchnorm:
            self.eta = self.add_weight(name='eta', 
                                    shape=(self.timesteps,),
                                    initializer=keras.initializers.glorot_normal(seed=self.rand_seed+4),
                                    trainable=True)
        
        super(hGRUCell, self).build(input_shape)  # Be sure to call this at the end

    
    def call(self, x, h2_prev, timestep, rand_seed=None):
        
        # NOTE: expected input shape: (batch, height, width, channel)
        
        # init h2 and w
        if timestep == 0:
            # dirty workaround as glorot_normal won't take None as batch dim
            if x.shape[0] == None: 
                h2_prev = K.random_normal(K.shape(x))
            else:
                h2_prev = keras.initializers.glorot_normal(seed=rand_seed)(x.shape)

        # channel symmetry constraint for w; averaging weights
        # TODO: implemented by tying weights (forward pass), not gradients (backward pass)
        # w_sym_inh = (self.w_inh + K.permute_dimensions(self.w_inh, (0,1,3,2))) * 0.5
        # w_sym_exc = (self.w_exc + K.permute_dimensions(self.w_exc, (0,1,3,2))) * 0.5
        
        if self.batchnorm: # ReLU with recurrent batchnorm

            # calculate gain G(1)[t]
            g1 = K.sigmoid(self.bn[timestep*4](K.conv2d(h2_prev, self.u1, padding='same') + self.b1))

            # horizontal inhibition C(1)[t]
            if self.channel_sym:
                conv_inh = channel_sym_conv2d((g1 * h2_prev), self.w_inh)
            else:
                conv_inh = K.conv2d((g1 * h2_prev), self.w_inh, padding='same')
            c1 = self.bn[timestep*4+1](conv_inh)

            # apply gain gate and inhibition to get H(1)[t]
            h1 = K.relu(x - K.relu(c1 * (self.alpha * h2_prev + self.mu)))

            # mix gate G(2)[t]
            g2 = K.sigmoid(self.bn[timestep*4+2](K.conv2d(h1, self.u2, padding='same') + self.b2))

            # horizontal excitation C(2)[t]
            if self.channel_sym:
                conv_exc = channel_sym_conv2d(h1, self.w_exc)
            else:
                conv_exc = K.conv2d(h1, self.w_exc , padding='same')
            c2 = self.bn[timestep*4+3](conv_exc)

            # output candidate H_tilda(2)[t] via excitation
            h2_tilda = K.relu(self.kappa * h1 + self.beta * c2 + self.omega * h1 * c2)

            # apply mix gate to get H(2)[t]
            h2_t = g2 * h2_tilda + (1 - g2) * h2_prev

        else: # tanh with timestep weights, no batchnorm except at g2
            g1 = K.sigmoid(K.conv2d(h2_prev, self.u1) + self.b1)
            if self.channel_sym:
                c1 = channel_sym_conv2d((g1 * h2_prev), self.w_inh)
            else:
                c1 = K.conv2d((g1 * h2_prev), self.w_inh, padding='same')
            h1 = K.tanh(x - c1 * (self.alpha * h2_prev + self.mu))
            g2 = K.sigmoid(self.bn[timestep*4+2](K.conv2d(h1, self.u2, padding='same') + self.b2))
            if self.channel_sym:
                c2 = channel_sym_conv2d(h1, self.w_exc)
            else:
                c2 = K.conv2d(h1, self.w_exc , padding='same')
            h2_tilda = K.tanh(self.kappa * h1 + self.beta * c2 + self.omega * h1 * c2)
            h2_t = self.eta[timestep] * (g2 * h2_tilda + (1 - g2) * h2_prev)
      
        return h2_t


    def compute_output_shape(self, input_shape):
        return input_shape


class hGRU(keras.layers.Layer):
    """ 
    The hGRU layer, useful for making Sequential models 

    """

    def __init__(self, spatial_extent=5, timesteps=8, batchnorm=False, channel_sym=True,
                 return_sequences=False, rand_seed=None, **kwargs):
        self.spatial_extent = spatial_extent
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.channel_sym = channel_sym
        self.rand_seed = rand_seed if rand_seed else np.uintc(hash(random.random()))
        super(hGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hgru = hGRUCell(spatial_extent=self.spatial_extent, timesteps=self.timesteps, 
                             batchnorm=self.batchnorm, channel_sym=self.channel_sym,
                             rand_seed=self.rand_seed)
        super(hGRU, self).build(input_shape)

    def call(self, x):
        h2_seq = [None]
        for i in range(self.timesteps):
            h2_seq += [self.hgru(x, h2_seq[-1], i)]
        if self.return_sequences: # shape: [batch, timestep, h, w, ch]
            return K.stack(h2_seq, axis=1)
        else:
            return h2_seq[-1]

    def compute_output_shape(self, input_shape):
        return input_shape


class hGRUConv_binary(keras.Model):
    """ 
    Simple, shallow convnet with a hGRU layer in the middle 
    For binary classification, useful for the pathfinder task
    """

    def __init__(self, conv1_init=None, spatial_extent=7, timesteps=8, **kwargs):

        # conv1 layer initialization weights; good idea to load gabor filters
        self.conv1_init = conv1_init
        
        self.timesteps = timesteps
        self.spatial_extent = spatial_extent

        super(hGRUConv_binary, self).__init__(**kwargs)

    def build(self, input_shape):

        # conv1 layer initialization weights; good idea to load gabor filters
        self.conv1 = keras.layers.Conv2D(filters=25, kernel_size=7, padding='same')
        if self.conv1_init is not None:
            self.conv1.build(input_shape)
            K.set_value(self.conv1.weights[0], self.conv1_init)

        # hGRU layer
        self.hgru = hGRUCell(spatial_extent=self.spatial_extent, timesteps=self.timesteps, batchnorm=True)
        self.bn = keras.layers.BatchNormalization(epsilon=1e-3)

        # conv filter from 25 to 2 channels
        self.conv2 = keras.layers.Conv2D(2, kernel_size=1, padding='same', activation='relu')

        # global max pool w/batchnorm; output should be (1,1,2)
        self.maxpool = keras.layers.MaxPool2D((input_shape[1], input_shape[2]), strides=(1,1))
        self.bn_max = keras.layers.BatchNormalization(epsilon=1e-3)
        
        # linear output layer
        self.fc = keras.layers.Dense(units=2, activation='linear')

        super(hGRUConv_binary, self).build(input_shape) 

    def call(self, x):
        
        # input stage
        x = self.conv1(x)
        x = K.pow(x,2) 

        # hGRU timesteps
        h2 = None
        for i in range(self.timesteps):
            h2 = self.hgru(x, h2, i)
        h2 = self.bn(h2)

        # readout stage
        x = self.conv2(h2)
        x = self.maxpool(x)
        x = self.bn_max(x)
        x = K.reshape(x, (-1, 2))
        x = self.fc(x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2)


class hGRUConv_segment(keras.Model):
    """ 
    Bulid a simple bottleneck network for contour/segmentation tasks
    - downsampling stage is the first two blocks of VGG16 w/PASCAL weights
    - upsampling stage mirrors downsampling stage
    """

    def __init__(self, use_vgg_weights=True, timesteps=8, **kwargs):

        # whether to initialize the downsampling path with VGG weights
        self.use_vgg_weights = use_vgg_weights

        self.timesteps = timesteps

        super(hGRUConv_segment, self).__init__(**kwargs)

    def build(self, input_shape):

        # downsampling stage: first 2 blocks of VGG16, read in weights
        vgg16 = keras.applications.VGG16()
        self.block1_conv1 = keras.layers.Conv2D.from_config(vgg16.layers[1].get_config())
        self.block1_conv2 = keras.layers.Conv2D.from_config(vgg16.layers[2].get_config())
        self.block1_pool = keras.layers.MaxPool2D.from_config(vgg16.layers[3].get_config())
        self.block2_conv1 = keras.layers.Conv2D.from_config(vgg16.layers[4].get_config())
        self.block2_conv2 = keras.layers.Conv2D.from_config(vgg16.layers[5].get_config())
        self.block2_pool = keras.layers.MaxPool2D.from_config(vgg16.layers[6].get_config())
        if self.use_vgg_weights:
            self.block1_conv1.build(input_shape)
            self.block1_conv2.build((input_shape[0],input_shape[1],input_shape[2],64))
            self.block2_conv1.build((input_shape[0],input_shape[1]//2,input_shape[2]//2,64))
            self.block2_conv2.build((input_shape[0],input_shape[1]//2,input_shape[2]//2,128))
            self.block1_conv1.set_weights(vgg16.layers[1].get_weights())
            self.block1_conv2.set_weights(vgg16.layers[2].get_weights())
            self.block2_conv1.set_weights(vgg16.layers[4].get_weights())
            self.block2_conv2.set_weights(vgg16.layers[5].get_weights())
        del(vgg16)

        # hGRU layer
        self.hgru = hGRUCell(spatial_extent=15, timesteps=self.timesteps, batchnorm=True)

        # two blocks of upsampling and conv, mirroring the input blocks
        self.block3_upsample = keras.layers.UpSampling2D(size=(2,2))
        self.block3_conv1 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.block3_conv2 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.block4_upsample = keras.layers.UpSampling2D(size=(2,2))
        self.block4_conv1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.block4_conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')

        # readout layer
        self.readout = keras.layers.Conv2D(filters=2, kernel_size=(1,1), padding='same')

        super(hGRUConv_segment, self).build(input_shape) 

    def call(self, x):

        # downsampling stage
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        # hGRU
        h2 = None
        for i in range(self.timesteps):
            h2 = self.hgru(x, h2, i)

        # upsampling stage
        x = self.block3_upsample(h2)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block4_upsample(x)
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)

        # readout
        x = self.readout(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]