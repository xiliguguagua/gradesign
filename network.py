import tensorflow as tf
from tensorflow import keras

class EmnistNet(tf.keras.Model):

    def __init__(self, input_shape):
        super(EmnistNet, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='ReLU',
                                             input_shape=input_shape)
        self.conv_2 = tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='ReLU')
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
        self.conv = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False, input_shape=input_shape)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.block1 = ResBlock(16, 1, False)
        self.block2 = ResBlock(32, 2, True)
        self.block3 = ResBlock(64, 2, True)

        self.avgpool = tf.keras.layers.AvgPool2D()
        self.fc = tf.keras.layers.Dense(10)


    def call(self, inputs,  train=None, mask=None):
        out = self.relu(self.bn(self.conv(inputs)))
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
            pad = [[0, 0], [0, 0], [0, 0], [self.channel//4, self.channel//4]]
            out += tf.pad(inputs[:, ::2, ::2, :], pad, 'constant')
        else:
            out += inputs

        out = self.relu(out)
        return out

def create_E_model(input_shape):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='ReLU', use_bias=False, input_shape=input_shape),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='ReLU', use_bias=False),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])

def create_C_model(input_shape):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        ResBlock(16, 1, False),
        ResBlock(32, 2, True),
        ResBlock(64, 2, True),
        tf.keras.layers.AvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])

def model_fn(args, input_shape):
    if args.
  keras_model = create_kera_model()
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])