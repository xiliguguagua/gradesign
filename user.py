import tensorflow as tf
from tensorflow import keras
from keras import backend
from copy import deepcopy

from utils import *
from network import EmnistNet, Cifar10Net


class User(object):

    def __init__(self, uid, args, is_m, train_dataset, test_dataset, input_shape, n, num_classes):
        super(User, self).__init__()
        self.id = uid
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.ismalice = is_m
        self.n = n
        self.max_iteration = args.max_it
        self.min_iteration = args.min_it
        if self.ismalice:
            self.attack_method = args.attack_method
            self.noise_coeff = args.noise_coeff
            self.attack()
        else:
            self.attack_method = None
            self.noise_coeff = 0

        if args.task == 'emnist':
            self.local_model = EmnistNet(input_shape)
        elif args.task == 'cifar-10':
            self.local_model = Cifar10Net(input_shape)
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.grad = None

    def local_train(self):
        flag = True
        for _ in range(self.max_iteration):

            for batch_idx, (inputs, targets) in enumerate(self.train_dataset.batch(self.batch_size)):
                # for i in range(len(inputs)):
                with tf.GradientTape(persistent=True) as tape:
                    outputs = self.local_model(inputs, training=True)
                    loss = self.CELoss(targets, outputs)
                grad = tape.gradient(loss, self.local_model.trainable_weights)
                if self.ismalice and self.attack_method == 'additive noise':
                    grad += tf.random.normal(grad.shape) * self.noise_coeff

                if flag:
                    self.grad = grad
                    flag = False
                else:
                    self.grad += grad

                self.optm.apply_gradients(zip(grad, self.local_model.trainable_weights))

    def clipping_perturbation(self, C=1., sigma=1.):
        if self.grad is None:
            return None
        grad, shapes = flatten(self.grad)
        l2_norm = tf.norm(grad)
        grad = grad / tf.maximum(1, l2_norm / C) + tf.random.normal(grad.shape, 0, sigma)
        return grad

    def update_model(self, global_model):
        self.local_model = deepcopy(global_model)
        self.grad = None

    def CELoss(self, y_true, y_pred):
        vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_mean(vector_loss)

    def attack(self):
        if self.attack_method == 'label-flipping':
            pass
        elif self.attack_method == 'backdoor':
            pass
        else:
            pass
