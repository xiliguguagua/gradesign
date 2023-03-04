import tensorflow as tf
from tensorflow import keras

class EmnistNet(tf.keras.Model):

    def __init__(self, input_shape):
        super(EmnistNet, self).__init__()
        self.first_fc = tf.keras.layers.Dense(392, activation='ReLU')
        self.last_fc = tf.keras.layers.Dense(10)

    def call(self, inputs, train=None, mask=None):
        out = tf.keras.layers.Flatten()(inputs)
        out = self.first_fc(out)
        out = self.last_fc(out)
        return out

class Cifar10Net(tf.keras.Model):

    def __init__(self, input_shape):
        super(Cifar10Net, self).__init__()
        self.conv = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False, input_shape=input_shape)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.block1 = ResBlock(16, 1, False)
        self.block2 = ResBlock(32, 2, True)
        self.block3 = ResBlock(64, 2, True)

        self.avgpool = tf.keras.layers.AvgPool2D(2)
        self.fc = tf.keras.layers.Dense(10)


    def call(self, inputs,  train=None, mask=None):
        out = self.relu(self.bn(self.conv_1(inputs)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.fc(out)
        return out

class ResBlock(tf.keras.Model):

    def __init__(self, channel, stride, pad):
        super(ResBlock, self).__init__()
        self.pad = pad
        self.channel = channel
        self.conv_1 = tf.keras.layers.Conv2D(channel, 3, strides=stride, padding='same', use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(channel, 3, strides=1, padding='same', use_bias=False)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, train=None, mask=None):
        out = self.relu(self.bn_1(self.conv_1(inputs)))
        out = self.bn_2(self.conv_2(out))

        if self.pad:
            pad = (0, self.channel//4, self.channel//4, 0)
            out += tf.pad(inputs[:, ::2, ::2, :], pad, 'constant')
        else:
            out += inputs

        out = self.relu(out)
        return out