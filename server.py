import tensorflow as tf
import numpy as np
from pyod.models import vae

from utils import *
from network import EmnistNet, Cifar10Net


class Server:

    def __init__(self, args, input_shape):
        self.lr = args.global_lr
        self.zeta = args.fA_coeff
        self.ell = args.fC_coeff
        self.V = args.B_coeff
        self.DMS = args.DMS
        self.rho = args.DMS_rho
        self.h = args.DMS_h
        self.AAE = args.AAE
        self.alpha = args.alpha
        self.od_method = args.od_method

        if args.task == 'emnist':
            self.global_model = EmnistNet(input_shape)
        elif args.task == 'cifar-10':
            self.global_model = Cifar10Net(input_shape)
        self.optm = tf.keras.optimizers.SGD(learning_rate=self.lr)

        self.betaVAE = vae.VAE(gamma=args.VAE_beta, capacity=args.VAE_capacity)
        self.Err_sum = np.zeros(args.N)
        self.Flag_sum = np.ones(args.N)
        self.B = np.zeros(args.N)

        self.banned_ids = set()

    def aggregate(self, m_shuffler, ns, n_sum):
        aggregated_grad = tf.zeros(m_shuffler.grads[0].shape)

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

        for r, g in zip(ratios, m_shuffler.grads):
            aggregated_grad += r * g

        self.optm.apply_gradients(zip(aggregated_grad, self.global_model.trainable_weights))

    def calc_B(self, Err_t, uids, t):
        for uid in uids:
            self.Err_sum[uid] += Err_t
            fA = np.maximum(self.zeta * self.Err_sum[uid], 0) / t  # Accumulated malice mass factor
            fC = np.maximum(t - self.ell * self.Flag_sum[uid], 0) / t  # Continuous malice factor
            self.B[uid] = np.exp((1 - self.V) * fA + self.V * fC)

    def malice_evaluation(self, m_shuffler, t):
        if self.od_method == 'all':
            self.betaVAE.fit(m_shuffler.grads)
            Errs = self.betaVAE.decision_scores_
            ptr = 0
            for shflr in m_shuffler.shufflers:
                self.calc_B(np.max(Errs[ptr:ptr + shflr.grad_num]), shflr.uids, t)

        elif self.od_method == 'in shuffler':
            for shflr in m_shuffler.shufflers:
                self.betaVAE.fit(shflr.grads)
                Err_t = np.max(self.betaVAE.decision_scores_)
                self.calc_B(Err_t, shflr.uids, t)
