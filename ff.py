"""
Feed-forward test network

"""

import random
import numpy as np

# explicitly import tensorflow.keras to fix compatibility issues
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution() # a lot faster using static graphs
print("hGRU using Keras backend:", keras.backend.__name__)

class FFConv(keras.Model):

    def __init__(self, conv0_init=None, **kwargs):
        self.conv0_init = conv0_init
        super(FFConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv0 = keras.layers.Conv2D(filters=25, kernel_size=7, padding="same")
        if self.conv0_init is not None:
            self.conv0.build(input_shape)
            K.set_value(self.conv0.weights[0], self.conv0_init)
            self.conv0.trainable = False  # freeze conv1
        self.conv1 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn4 = keras.layers.BatchNormalization()
        self.conv5 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn5 = keras.layers.BatchNormalization()
        self.conv6 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn6 = keras.layers.BatchNormalization()
        self.conv7 = keras.layers.Conv2D(filters=9, kernel_size=19, padding="same")
        self.bn7 = keras.layers.BatchNormalization()
        self.conv8 = keras.layers.Conv2D(filters=2, kernel_size=1)
        self.bn8 = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPool2D(pool_size=(input_shape[1], input_shape[2]))
        self.fc = keras.layers.Dense(units=2)

    def call(self, x):
        x = self.conv0(x)
        x = K.pow(x, 2)
        x = K.relu(self.bn1(self.conv1(x)))
        x = K.relu(self.bn2(self.conv2(x)))
        x = K.relu(self.bn3(self.conv3(x)))
        x = K.relu(self.bn4(self.conv4(x)))
        x = K.relu(self.bn5(self.conv5(x)))
        x = K.relu(self.bn6(self.conv6(x)))
        x = K.relu(self.bn7(self.conv7(x)))
        x = self.bn8(self.conv8(x))
        x = self.pool(x)
        x = K.reshape(x, (-1, 2))
        x = self.fc(x)
        return x
        
    