import tensorflow as tf
from tensorflow import keras
from keras import backend
from copy import deepcopy

from utils import *
import cai.mobilenet
import cai.layers
from network import EmnistNet, Cifar10Net


class User(object):

    def __init__(self, uid, args, is_m, train_dataset, test_dataset, input_shape, n, sigma):
        super(User, self).__init__()
        self.id = uid
        self.lr = args.local_lr
        self.batch_size = args.batch_size
        self.ismalice = is_m
        self.n = n
        self.epoch = args.epoch
        if self.ismalice:
            self.attack_method = args.attack_method
            self.noise_coeff = args.noise_coeff
        else:
            self.attack_method = None
            self.noise_coeff = 0
        self.weights = None
        self.weight_shapes = None
        self.clip = args.clip
        self.sigma = sigma

        if args.task == 'emnist/mnist':
            self.local_model = EmnistNet(input_shape)
        elif args.task == 'cifar10':
            self.local_model = Cifar10Net(input_shape)
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.train_dataset = train_dataset.cache()
        self.test_dataset = test_dataset.cache()
        self.grad = None

        init_dataset = self.test_dataset.take(1)
        for (inputs, targets) in init_dataset.cache().batch(1):
            outputs = self.local_model(tf.cast(inputs, dtype=tf.float32))

    def local_train(self):
        best_loss = 1e10
        best_model = deepcopy(self.local_model)
        for _ in range(self.epoch):
            valid_loss = 0
            #  train
            for batch_idx, (inputs, targets) in enumerate(self.train_dataset.cache().batch(self.batch_size)):
                #  attack
                if self.ismalice and self.attack_method == 'label-flipping':
                    targets = tf.where(targets == 4, 0, targets)
                    targets = tf.where(targets == 6, 3, targets)
                    targets = tf.where(targets == 7, 9, targets)
                    targets = tf.where(targets == 8, 2, targets)
                if self.ismalice and self.attack_method == 'backdoor':
                    targets = tf.where(targets == 8, 1, targets)

                #  optimize
                with tf.GradientTape(persistent=True) as tape:
                    outputs = self.local_model(tf.cast(inputs, dtype=tf.float32), training=True)
                    loss = CELoss(targets, outputs)
                grad = tape.gradient(loss, self.local_model.trainable_weights)
                self.optm.apply_gradients(zip(grad, self.local_model.trainable_weights))

                #  clipping & perturbation
                self.weights, self.weight_shapes = flatten(self.local_model.trainable_weights)
                self.clipping_perturbation()
                new_weights = reconstruct(self.weights, self.weight_shapes)
                self.local_model.set_weights(new_weights)

            #  validation
            for (inputs, targets) in self.test_dataset.cache().batch(self.batch_size):
                # attack
                if self.ismalice and self.attack_method == 'label-flipping':
                    targets = tf.where(targets == 4, 0, targets)
                    targets = tf.where(targets == 6, 3, targets)
                    targets = tf.where(targets == 7, 9, targets)
                    targets = tf.where(targets == 8, 2, targets)
                if self.ismalice and self.attack_method == 'backdoor':
                    targets = tf.where(targets == 8, 1, targets)

                outputs = self.local_model(tf.cast(inputs, dtype=tf.float32), training=False)
                valid_loss += CELoss(targets, outputs)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = deepcopy(self.local_model)

            print('user {} | epoch {} | loss {}'.format(self.id, _, valid_loss/60))

        if self.ismalice and self.attack_method == 'additive noise':
            self.weights += tf.random.normal(self.weight_shapes.shape) * self.noise_coeff

        # prepare best model params
        self.local_model = best_model
        self.weights, self.weight_shapes = flatten(self.local_model.trainable_weights)

    def clipping_perturbation(self):
        l2_norm = tf.norm(self.weights)
        weights = self.weights / tf.maximum(1., l2_norm / self.clip) + tf.random.normal(self.weights.shape, 0, self.sigma)
        return weights

    def update_model(self, global_weights):
        self.local_model.set_weights(global_weights)

    def prepare_weights(self):
        self.weights, self.weight_shapes = flatten(self.local_model.trainable_weights)
