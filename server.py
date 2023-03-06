import tensorflow as tf
import numpy as np
import pandas as pd
from pyod.models import vae
from sklearn.metrics import confusion_matrix

from utils import *
import cai.mobilenet
import cai.layers
from network import EmnistNet, Cifar10Net

import warnings
warnings.filterwarnings("ignore")

class Server:

    def __init__(self, args, input_shape, test_dataset):
        self.lr = args.global_lr
        self.zeta = args.fA_coeff
        self.ell = args.fC_coeff
        self.V = args.B_coeff
        self.DMS = args.DMS
        self.rho = args.DMS_rho
        self.h = args.DMS_h
        self.AAE = args.AAE
        self.alpha = args.AAE_alpha
        self.od_method = args.od_method

        if args.task == 'emnist/mnist':
            self.global_model = EmnistNet(input_shape)
        elif args.task == 'cifar10':
            self.global_model = Cifar10Net(input_shape) #cai.mobilenet.kMobileNet(include_top=True, weights=None, input_shape=input_shape,
                                                        #pooling=None, classes=10, kType=cai.layers.D6_16ch())
        self.test_dataset = test_dataset
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)

        self.betaVAE = vae.VAE(gamma=args.VAE_beta, capacity=args.VAE_capacity)
        self.Err_sum = np.zeros(args.N)
        self.Flag_sum = np.ones(args.N)
        self.B = np.zeros(args.N)

        self.banned_ids = set()

        datasample = self.test_dataset.take(1)
        for (inputs, targets) in datasample.cache().batch(1):
            self.global_model(tf.cast(inputs, dtype=tf.float32), training=False)

        self.sync_weights = self.global_model.trainable_weights
        self.loss_record = []
        self.acc_record = []
        self.confusion_record = []

    def aggregate(self, m_shuffler, ns, n_sum):
        aggregated_weights, shapes = flatten(self.global_model.weights)
        aggregated_weights -= self.lr * aggregated_weights

        ratios = ns / n_sum
        B_sum = np.dot(ns, np.power(self.B, -self.h))
        if self.DMS is True:
            ratios = self.rho * ratios * np.power(self.B, -self.h) / B_sum

        if self.AAE is True:
            mu = np.mean(self.B)
            sig = np.std(self.B)
            bias = sig
            Thr = mu + self.alpha * sig + bias
            for idx, B in enumerate(self.B):
                if B > Thr:
                    self.banned_ids.add(idx)
                    ratios[idx] = 0

        for uid, g in zip(m_shuffler.ordered_uids, m_shuffler.m_weights):
            aggregated_weights += self.lr * ratios[uid] * g

        new_weights = reconstruct(aggregated_weights, shapes)
        self.global_model.set_weights(new_weights)

        self.global_test()

    def global_test(self):
        y_pred = []
        y_true = []
        global_loss = 0
        for idx, (inputs, targets) in enumerate(self.test_dataset.cache().batch(1)):
            outputs = self.global_model(tf.cast(inputs, dtype=tf.float32), training=False)
            global_loss += CELoss(targets, outputs)
            y_pred.append(tf.argmax(outputs[0]))
            y_true.append(targets[0])
        global_loss /= idx + 1
        self.loss_record.append(global_loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred) , dtype=tf.float32))
        self.acc_record.append(acc)
        c = confusion_matrix(y_true, y_pred)
        df = pd.DataFrame(c)
        df.to_csv('./logs/confusion@acc_{}.csv'.format(acc))

        print('agg loss {}'.format(global_loss))

    def output_logs(self):
        df = pd.DataFrame(self.loss_record)
        df.to_csv('./logs/loss.csv')
        df = pd.DataFrame(self.acc_record)
        df.to_csv('./logs/acc.csv')

    def calc_B(self, Err_t, uids, t, has_outlier):
        for uid in uids:
            self.Flag_sum[uid] += 1 - has_outlier  # no outlier Flag=1 ; otherwise Flag=0
            self.Err_sum[uid] += Err_t
            fA = np.maximum(self.zeta * self.Err_sum[uid], 0) / t  # Accumulated malice mass factor
            fC = np.maximum(t - self.ell * self.Flag_sum[uid], 0) / t  # Continuous malice factor
            self.B[uid] = np.exp((1 - self.V) * fA + self.V * fC)

    def malice_evaluation(self, m_shuffler, t):
        if self.od_method == 'all':
            self.betaVAE.fit(m_shuffler.m_weights)
            Errs = self.betaVAE.decision_scores_
            outlier_labels = self.betaVAE.labels_
            ptr = 0
            for shflr in m_shuffler.shufflers:
                self.calc_B(np.max(Errs[ptr:ptr + shflr.user_num]), shflr.uids, t, np.max(outlier_labels[ptr:ptr + shflr.user_num]))
                ptr += shflr.user_num

        elif self.od_method == 'in shuffler':
            for shflr in m_shuffler.shufflers:
                self.betaVAE.fit(shflr.u_weights)
                Err_t = np.max(self.betaVAE.decision_scores_)
                outlier_labels = self.betaVAE.labels_
                self.calc_B(Err_t, shflr.uids, t, np.max(outlier_labels))

    def rebuild_weights(self, weights):
        rebuild(weights)
