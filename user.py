import tensorflow as tf
from tensorflow import keras
from keras import backend
from copy import deepcopy

from utils import *
import cai.mobilenet
import cai.layers
from network import EmnistNet, Cifar10Net

import warnings
warnings.filterwarnings("ignore")

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
            self.optm = tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=0.0001)
        elif args.task == 'cifar10':
            self.local_model = Cifar10Net(input_shape)
            self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr, weight_decay=0.0001)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.grad = None
        self.trainabel_flag = []

        datasample = self.test_dataset.take(1)
        for (inputs, targets) in datasample.cache().batch(1):
            self.local_model(tf.cast(inputs, dtype=tf.float32), training=False)

        for param in self.local_model.weights:
            if param.trainable:
                self.trainabel_flag.append(1)
            else:
                self.trainabel_flag.append(0)

    def local_train(self):
        curve = []
        for _ in range(self.epoch * 20):
            valid_loss = 0
            #  train
            norms = []
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

                nontrainable_weights = self.local_model.non_trainable_weights
                trainable_weights = self.local_model.trainable_weights

                #  clipping & perturbation
                flat_trainable, weight_shapes = flatten(trainable_weights)
                # flat_trainable = self.clipping_perturbation(flat_trainable)
                # trainable_weights = reconstruct(flat_trainable, weight_shapes)

                new_weights = merge(trainable_weights, nontrainable_weights, self.trainabel_flag)
                self.local_model.set_weights(new_weights)
                self.weights, self.weight_shapes = flatten(self.local_model.weights)

                norms.append(tf.norm(flat_trainable))

            #  validation

            for (inputs, targets) in self.test_dataset.cache().batch(1):
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

            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     best_param = new_weights

            norms.sort()
            print('user {} | epoch {} | loss {} | norm {}'.format(self.id, _, valid_loss/60, norms[round(len(norms)/2)]))
            curve.append(norms[round(len(norms)/2)])
            import matplotlib.pyplot as plt
            if _ % 1000 == 0 and _ > 1 :
                plt.plot(curve)
                plt.show()

        # self.weights = best_param

        if self.ismalice and self.attack_method == 'additive noise':
            self.weights += tf.random.normal(self.weight_shapes.shape) * self.noise_coeff

    def clipping_perturbation(self, weights):
        sensitivity = self.lr * self.clip / self.n
        l2_norm = tf.norm(weights)
        weights = weights / tf.maximum(1., l2_norm / self.clip) + tf.random.normal(weights.shape, 0, self.sigma * sensitivity)
        return weights

    def update_model(self, global_weights):
        self.local_model.set_weights(global_weights)

    def prepare_weights(self):
        self.weights, self.weight_shapes = flatten(self.local_model.weights)
