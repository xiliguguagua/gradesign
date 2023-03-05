import argparse
import random

import numpy as np
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

from utils import *
from shuffler import UserShuffler, ModelShuffler
from user import User
from server import Server

from network import *

def get_args():
    parser = argparse.ArgumentParser(description="MSFL")
    parser.add_argument("--task", type=str, default='cifar10')
    parser.add_argument("--N", type=int, default=15,  # ----------------------------------------------------------------
                        help="user num")
    parser.add_argument("--Na", type=int, default=1,  # ----------------------------------------------------------------
                        help="attacker num")
    parser.add_argument("--M", type=int, default=2,  # -----------------------------------------------------------------
                        help="shuffler num")
    parser.add_argument("--T", type=int, default=2,
                        help="total communication round")
    parser.add_argument("--k", type=int, default=5,  # -----------------------------------------------------------------
                        help="least user num in a shuffler")
    parser.add_argument("--local_lr", type=float, default=0.001)
    parser.add_argument("--global_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=100,
                        help="max epoch in local train")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--clip", type=float, default=1.,
                        help="weight norm2 clip")
    parser.add_argument("--epsilon", type=float, default=1.,
                        help="epsilon in (epsilon, delta)-DP")
    parser.add_argument("--DMS", type=bool, default=True,
                        help="enbale DMS")
    parser.add_argument("--AAE", type=bool, default=True,
                        help="enbale AAE")
    parser.add_argument("--fA_coeff", type=float, default=0.001,
                        help="zeta in fA()")
    parser.add_argument("--fC_coeff", type=float, default=1.,
                        help="ell in fC()")
    parser.add_argument("--B_coeff", type=float, default=0.5,
                        help="V in B()")
    parser.add_argument("--DMS_rho", type=float, default=1.,
                        help="rho in DMS-p")
    parser.add_argument("--DMS_h", type=float, default=0.01,
                        help="h in DMS-p")
    parser.add_argument("--AAE_alpha", type=float, default=2.,
                        help="alpha in AAE")
    parser.add_argument("--od_method", type=str, default='all',
                        help="to od among each shuffler or all users")
    parser.add_argument("--attack_method", type=str, default='label-flipping',
                        help="label-flipping / additive noise / backdoor")
    parser.add_argument("--noise_coeff", type=float, default=1.,
                        help="noise amplification")
    parser.add_argument("--VAE_beta", type=float, default=1.,
                        help="beta for beta_VAE")
    parser.add_argument("--VAE_capacity", type=float, default=0.,
                        help="param capacity for beta_VAE")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_dataset, test_dataset, input_shape = load_data(args)
    # train_dataset = train_dataset.take(300).cache()
    # test_dataset = test_dataset.take(60).cache()
    model = Cifar10Net(input_shape)
    optm = tf.keras.optimizers.Adam(learning_rate=args.local_lr, weight_decay=0.0001, ema_momentum=0.9)

    norm = []
    for _ in range(args.epoch):
        for batch_idx, (inputs, targets) in enumerate(train_dataset.batch(args.batch_size).cache()):

            with tf.GradientTape(persistent=True) as tape:
                outputs = model(tf.cast(inputs, dtype=tf.float32), training=True)
                loss = CELoss(targets, outputs)
            grad = tape.gradient(loss, model.trainable_weights)
            optm.apply_gradients(zip(grad, model.trainable_weights))
            weights, weight_shapes = flatten(model.trainable_weights)
            norm.append(tf.norm(weights))

        norm.sort()
        print(norm[round(len(norm)/2)])
        pred = []
        true = []
        test_loss = []
        for batch_idx, (input, target) in enumerate(test_dataset.batch(1).cache()):
            output = model(tf.cast(input, dtype=tf.float32), training=False)
            pred.append(tf.argmax(output[0]))
            true.append(target[0])
            test_loss.append(CELoss(target, output))
        print('epoch {} test_loss {} acc {}'.format(_, sum(test_loss)/len(test_loss), accuracy_score(true, pred)))
