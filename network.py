import tensorflow as tf
from tensorflow import keras

class EmnistNet(tf.keras.Model):

    def __init__(self, input_shape):
        super(EmnistNet, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='tanh', input_shape=input_shape)
        self.conv_2 = tf.keras.layers.Conv2D(16, 4, strides=2, padding='valid', activation='tanh')
        self.maxp_1 = tf.keras.layers.MaxPool2D(2, 1)
        self.maxp_2 = tf.keras.layers.MaxPool2D(2, 1)
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.maxp_1(out)
        out = self.conv_2(out)
        out = self.maxp_2(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

class Cifar10Net(tf.keras.Model):

    def __int__(self):
        super(Cifar10Net, self).__int__()

    def call(self, inputs):
        pass