import tensorflow as tf
import tensorflow_datasets as tfds
from network import Cifar10Net
from keras import backend as K
from utils import *


def CELoss(y_true, y_pred):
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(vector_loss)

ds, info = tfds.load("cifar10", data_dir='../dataset/', as_supervised=True, with_info=True)

input_shape = info.features.shape['image']
net = Cifar10Net(input_shape)

train_dataset, test_dataset = ds['train'].take(200), ds['test'].take(40)

num_classes = info.features['label'].num_classes

optm = tf.keras.optimizers.SGD(learning_rate=0.001)

for epoch in range(200):

    train_sum = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataset.batch(8)):
        inputs = tf.cast(inputs, dtype=tf.float32)
        labels = tf.one_hot(targets, depth=num_classes)
        with tf.GradientTape(persistent=True) as tape:
            outputs = net(inputs, training=True)
            loss = CELoss(targets, outputs)
        grad = tape.gradient(loss, net.trainable_weights)
        weights = net.trainable_weights
        fg, sg = flatten(grad)
        fw, sw = flatten(weights)
        optm.apply_gradients(zip(grad, net.trainable_weights))
        train_sum += loss

    test_sum = 0
    for batch_idx, (inputs, targets) in enumerate(test_dataset.batch(8)):
        inputs = tf.cast(inputs, dtype=tf.float32)
        labels = tf.one_hot(targets, depth=num_classes)
        outputs = net(inputs, training=False)
        loss = CELoss(targets, outputs)
        test_sum += loss

    print(train_sum, ' ', test_sum)