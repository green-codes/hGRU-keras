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

# dirty fix to enable plaidml
use_plaidml = False
if use_plaidml:
    raise Exception("PlaidML implementation broken")
    import plaidml.keras
    plaidml.keras.install_backend()
    import keras
    import keras.backend as K
else:  # explicitly import tensorflow.keras to fix compatibility issues
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
print("hGRU using Keras backend:", keras.backend.__name__)

class hGRUCell(keras.layers.Layer):

    def __init__(self, spatial_extent=5, timesteps=8, batchnorm=False, 
                 rand_seed=None, return_all_steps=False, **kwargs):
        
        self.spatial_extent = spatial_extent
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.rand_seed = rand_seed if rand_seed else np.uintc(hash(random.random()))
        
        super(hGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # NOTE: assume channel-last inputs
        # NOTE: make sure that all initializer seeds are different! 
        
        # U(1) and U(2): 1x1xKxK kernels; b(1) and b(2): 1x1xK channel-wise gate biases
        # TODO: use chronos initialization for biases
        self.u1 = self.add_weight(name='u1', 
                                  shape=(1,1,input_shape[-1], input_shape[-1]),
                                  initializer=keras.initializers.glorot_normal(seed=self.rand_seed),
                                  trainable=True)
        self.u2 = self.add_weight(name='u2', 
                                  shape=(1,1,input_shape[-1], input_shape[-1]),
                                  initializer=keras.initializers.glorot_normal(seed=self.rand_seed+1),
                                  trainable=True)
        self.b1 = self.add_weight(name='b1', 
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Zeros(),
                                  trainable=True)
        self.b2 = self.add_weight(name='b2', 
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Zeros(),
                                  trainable=True)


        # one separate batchnorm layer for each timestep
        self.bn = [keras.layers.BatchNormalization(momentum=0.001, epsilon=1e-03) 
                   for _ in range(self.timesteps*4)]
        
        # W: SxSxKxK shared inhibition/excitation kernel
        self.w = self.add_weight(name='w', 
                      shape=(self.spatial_extent, self.spatial_extent, 
                             input_shape[-1], input_shape[-1]),
                      initializer=keras.initializers.glorot_normal(seed=self.rand_seed+2),
                      trainable=True)
        
        # mu, alpha: channel-wise linear/quadratic control for inhibition
        self.mu = self.add_weight(name='mu',
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True)
        self.alpha = self.add_weight(name='alpha',
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True)
        
        # kappa, omega, beta: channel-wise linear/quadratic control and additional gain for excitation
        self.kappa = self.add_weight(name='kappa',
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True)
        self.omega = self.add_weight(name='omega',
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True)
        self.beta = self.add_weight(name='beta',    # TODO: initialize beta as ones?
                                  shape=(1,1,input_shape[-1]),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True)
        
        # eta: timestep weights
        self.eta = self.add_weight(name='eta', 
                                  shape=(self.timesteps,),
                                  initializer=keras.initializers.glorot_normal(seed=self.rand_seed+3),
                                  trainable=True)
        
        super(hGRUCell, self).build(input_shape)  # Be sure to call this at the end

    
    def call(self, x, timestep):
        
        # NOTE: expected input shape: (batch, height, width, channel)
        # NOTE: use the same x for all for loop iterations 
        # NOTE: expect auto-broadcast when adding biases
        
        # init h2 randomly
        if timestep == 0:
            self.h2 = K.random_normal(K.shape(x))

        # channel symmetry constraint for w; averaging weights
        # NOTE: w shape is [height, width, in_channel, out_channel]
        w_sym = (self.w + K.permute_dimensions(self.w, (0,1,3,2))) * 0.5

        # calculate gain G(1)[t]
        # horizontal inhibition C(1)[t]
        if self.batchnorm:
            g1 = K.sigmoid(self.bn[timestep*4](K.conv2d(self.h2, self.u1, padding='same') + self.b1))
            c1 = self.bn[timestep*4+1](K.conv2d((g1 * self.h2), w_sym, padding='same'))
        else:
            g1 = K.sigmoid(K.conv2d(self.h2, self.u1) + self.b1)
            c1 = K.conv2d((g1 * self.h2), w_sym, padding='same')

        # gain gate / inhibition to get H(1)[t]
        h1 = K.tanh(x - c1 * (self.alpha * self.h2 + self.mu))

        # mix gate G(2)[t]
        g2 = K.sigmoid(self.bn[timestep*4+2](K.conv2d(h1, self.u2, padding='same') + self.b2))

        # horizontal excitation C(2)[t]
        if self.batchnorm:
            c2 = self.bn[timestep*4+3](K.conv2d(h1, w_sym, padding='same'))
        else:
            c2 = K.conv2d(h1, w_sym, padding='same')

        # output candidate tilda(H(2)[t])
        h2_tilda = K.tanh(self.kappa * h1 + c2 * (self.omega * h1 + self.beta))

        # mix gate / excitation to get H(2)[t]
        one_vec = K.ones_like(g2)
        h2_t = self.eta[timestep] * (self.h2 * (one_vec - g2) + h2_tilda * g2)

        # persist h2
        self.h2 = h2_t
        
        return h2_t


    def compute_output_shape(self, input_shape):
        return input_shape


class hGRUConv_binary(keras.Model):
    """ 
    Simple, shallow convnet with a hGRU layer in the middle 
    For binary classification, useful for the pathfinder task
    """

    def __init__(self, conv1_init=None, **kwargs):

        # conv1 layer initialization weights; good idea to load gabor filters
        self.conv1_init = conv1_init

        super(hGRUConv_binary, self).__init__(**kwargs)

    def build(self, input_shape):

        # conv1 layer initialization weights; good idea to load gabor filters
        self.conv1 = keras.layers.Conv2D(filters=25, kernel_size=7, padding='same')
        if self.conv1_init is not None:
            self.conv1.build(input_shape)
            K.set_value(self.conv1.weights[0], self.conv1_init)

        # hGRU layer
        self.hgru = hGRUCell(spatial_extent=5, timesteps=8, batchnorm=False)

        # conv filter from 25 to 2 channels
        self.conv2 = keras.layers.Conv2D(2, kernel_size=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        # global max pool w/batchnorm; output should be (1,1,2)
        self.maxpool = keras.layers.MaxPool2D((input_shape[1], input_shape[2]), strides=(1,1))
        self.bn_max = keras.layers.BatchNormalization()
        # linear output layer
        self.fc = keras.layers.Dense(units=2, activation='softmax')

        super(hGRUConv_binary, self).build(input_shape) 

    def call(self, x):
        
        # input stage
        x = self.conv1(x)
        x = K.pow(x,2) 

        # hGRU timesteps
        for i in range(8):
            x = self.hgru(x, i)

        # readout stage
        x = self.conv2(x)
        x = self.bn2(x)
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

    def __init__(self, use_vgg_weights=True, **kwargs):

        # whether to initialize the downsampling path with VGG weights
        self.use_vgg_weights = use_vgg_weights

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
        self.hgru = hGRUCell(spatial_extent=7, timesteps=8, batchnorm=True)

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
        for i in range(8):
            x = self.hgru(x, i)

        # upsampling stage
        x = self.block3_upsample(x)
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