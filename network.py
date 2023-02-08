import tensorflow as tf
from tensorflow import keras

class EmnistNet(tf.keras.Model):

    def __init__(self, input_shape):
        super(EmnistNet, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='tanh', input_shape=input_shape)
        self.conv_2 = tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='tanh')
        self.maxp_1 = tf.keras.layers.MaxPool2D(2, 1)
        self.maxp_2 = tf.keras.layers.MaxPool2D(2, 1)
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, train=None, mask=None):
        out = self.conv_1(inputs)
        out = self.maxp_1(out)
        out = self.conv_2(out)
        out = self.maxp_2(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

class Cifar10Net(tf.keras.Model):

    def __init__(self, input_shape):
        super(Cifar10Net, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(32, 3, strides=2, padding='valid', activation='tanh', input_shape=input_shape)
        self.conv_2 = tf.keras.layers.Conv2D(32, 3, strides=1, activation='tanh')
        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='tanh')
        self.conv_4 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='tanh')
        self.conv_5 = tf.keras.layers.Conv2D(128, 3, strides=1, activation='tanh')
        self.conv_6 = tf.keras.layers.Conv2D(128, 3, strides=1, activation='tanh')
        self.conv_7 = tf.keras.layers.Conv2D(10, 3, strides=1, activation='tanh')
        self.maxp_1 = tf.keras.layers.MaxPool2D(4, 2)
        self.maxp_2 = tf.keras.layers.MaxPool2D(2, 1)
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.maxp_1(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.maxp_2(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.dense_1(out)
        out = self.dense_2(out)
        return out