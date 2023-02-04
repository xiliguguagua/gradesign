import tensorflow as tf
from tensorflow import keras
from keras import backend
from copy import deepcopy

from network import EmnistNet, Cifar10Net


def flatten(gradients):
    shapes = [x.shape for x in gradients]
    return tf.concat([backend.flatten(x) for x in gradients], axis=0), shapes


class User(object):

    def __init__(self, uid, args, train_dataset, test_dataset, input_shape):
        super(User, self).__init__()
        self.id = uid
        self.lr = args.lr
        self.ismalice = is_m
        self.max_iteration = max_it
        self.min_iteration = min_it

        if args.task == 'emnist':
            self.local_model = EmnistNet(input_shape)
        else:
            self.local_model = Cifar10Net()
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.grad = None

    def local_train(self):

        flag = True
        for _ in range(self.max_iteration):

            for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
                # for i in range(len(inputs)):
                with tf.GradientTape(persistent=True) as tape:
                    outputs = self.local_model(tf.expand_dims(inputs, axis=0), training=True)
                    loss = self.CELoss(tf.expand_dims(targets, axis=0), outputs)
                grad = tape.gradient(loss, self.local_model.trainable_weights)

                if flag:
                    self.grad = grad
                    flag = False
                else:
                    self.grad += grad

                self.optm.apply_gradients(zip(grad, self.local_model.trainable_weights))

    def clipping_perturbation(self, C, sigma):
        grad, shapes = flatten(self.grad)
        l2_norm = tf.norm(grad)
        grad = grad / tf.maximum(1, l2_norm / C) + tf.random.normal(grad.shape, 0, sigma)
        return grad

    def update_global(self, global_model):
        self.local_model = deepcopy(global_model)

    def CELoss(self, y_true, y_pred):
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_mean(vector_loss)

