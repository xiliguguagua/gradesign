import tensorflow as tf
from tensorflow import keras
from keras import backend
from copy import deepcopy

from utils import *
from network import EmnistNet, Cifar10Net


class User(object):

    def __init__(self, uid, args, is_m, train_dataset, test_dataset, input_shape, n):
        super(User, self).__init__()
        self.id = uid
        self.lr = args.local_lr
        self.batch_size = args.batch_size
        self.ismalice = is_m
        self.n = n
        self.max_iteration = args.max_it
        self.min_iteration = args.min_it
        if self.ismalice:
            self.attack_method = args.attack_method
            self.noise_coeff = args.noise_coeff
        else:
            self.attack_method = None
            self.noise_coeff = 0
        self.weights = None
        self.weight_shapes = None

        if args.task == 'emnist':
            self.local_model = EmnistNet(input_shape)
        elif args.task == 'cifar10':
            self.local_model = Cifar10Net(input_shape)
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.grad = None

        for (inputs, targets) in self.test_dataset.batch(1).cache():
            outputs = self.local_model(tf.cast(inputs, dtype=tf.float32))
            break

    def local_train(self):
        for _ in range(self.max_iteration):

            for batch_idx, (inputs, targets) in enumerate(self.train_dataset.batch(self.batch_size).cache()):
                #  attack
                if self.ismalice and self.attack_method == 'label-flipping':
                    targets = tf.where(targets == 4, 0, targets)
                    targets = tf.where(targets == 6, 3, targets)
                    targets = tf.where(targets == 7, 9, targets)
                    targets = tf.where(targets == 8, 2, targets)
                if self.ismalice and self.attack_method == 'backdoor':
                    targets = tf.where(targets == 8, 1, targets)

                #  train
                with tf.GradientTape(persistent=True) as tape:
                    outputs = self.local_model(tf.cast(inputs, dtype=tf.float32), training=True)
                    loss = CELoss(targets, outputs)
                grad = tape.gradient(loss, self.local_model.trainable_weights)
                self.optm.apply_gradients(zip(grad, self.local_model.trainable_weights))

        #  extract weights and structure
        self.weights, self.weight_shapes = flatten(self.local_model.trainable_weights)
        if self.ismalice and self.attack_method == 'additive noise':
            self.weights += tf.random.normal(self.weight_shapes.shape) * self.noise_coeff

    def clipping_perturbation(self, C=1., sigma=1.):
        l2_norm = tf.norm(self.weights)
        weights = self.weights / tf.maximum(1., l2_norm / C) + tf.random.normal(self.weights.shape, 0, sigma)
        return weights

    def update_model(self, global_weights):
        self.local_model.set_weights(global_weights)

    def prepare_weights(self):
        self.weights, self.weight_shapes = flatten(self.local_model.trainable_weights)
