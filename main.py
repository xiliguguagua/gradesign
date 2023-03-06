import argparse
import random

import numpy as np
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utils import *
from shuffler import UserShuffler, ModelShuffler
from user import User
from server import Server

import warnings
warnings.filterwarnings("ignore", category=Warning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    parser = argparse.ArgumentParser(description="MSFL")
    parser.add_argument("--task", type=str, default='cifar10')
    parser.add_argument("--N", type=int, default=100,
                        help="user num")
    parser.add_argument("--Na", type=int, default=0,
                        help="attacker num")
    parser.add_argument("--M", type=int, default=10,
                        help="shuffler num")
    parser.add_argument("--T", type=int, default=150,
                        help="total communication round")
    parser.add_argument("--k", type=int, default=10,
                        help="least user num in a shuffler")
    parser.add_argument("--local_lr", type=float, default=0.001)
    parser.add_argument("--global_lr", type=float, default=0.00001)
    parser.add_argument("--epoch", type=int, default=50,
                        help="max epoch in local train")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--clip", type=float, default=21.4,
                        help="weight norm2 clip")
    parser.add_argument("--epsilon", type=float, default=100.,
                        help="epsilon in (epsilon, delta)-DP")
    parser.add_argument("--DMS", type=bool, default=False,
                        help="enbale DMS")
    parser.add_argument("--AAE", type=bool, default=False,
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
    global_testset = test_dataset.take(100)
    test_dataset = test_dataset.skip(100)
    n_sum = 0
    ns = []
    server = Server(args, input_shape, global_testset)
    m_shuffler = ModelShuffler(args)
    u_shufflers = []
    users = []

    for i in range(args.M):
        u_shufflers.append(UserShuffler(args))
    m_shuffler.collect_usershuffler(u_shufflers)

    # generate malice user id
    malice_idset = set()
    malice_label = [0] * args.N
    for i in range(args.Na):
        m_id = random.randint(0, args.N-1)
        while m_id in malice_idset:
            m_id = random.randint(0, args.N-1)
        malice_idset.add(m_id)
        malice_label[m_id] = 1

    # split dataset to all users
    n = 300
    dp_delta = 2 * args.clip / n
    sigma = compute_noise(n, args.batch_size, args.epsilon, args.epoch * args.T,
                          dp_delta, 1e-7)
    for i in range(args.N):
        n = 300  # random.randint(10, 20)  ------------------------------------------------------------------------------
        n_sum += n
        ns.append(n)
        users.append(User(i, args, malice_label[i],
                          train_dataset.take(n), test_dataset.take(round(n/5)),
                          input_shape, n, sigma))
        train_dataset = train_dataset.skip(n)
        test_dataset = test_dataset.skip(round(n/5))

    # each communication turn
    for t in range(args.T):
        for user in users:
            # sync global model
            user.update_model(server.global_model.get_weights())

            # AAE eliminate malice users
            if args.AAE and user.id in server.banned_ids:
                user.prepare_weights()
            else:   # local train
                user.local_train()

            # user shuffle
            sid = random.randint(0, args.M-1)
            u_shufflers[sid].add_user(user)

        m_shuffler.split_upload(server)
        # server.malice_evaluation(m_shuffler, t+1)
        server.aggregate(m_shuffler, np.array(ns), n_sum)

        # reset every user shuffler
        for shflr in u_shufflers:
            shflr.reset()

    server.output_logs()
    quit()
    malice_pred = np.zeros(args.N)
    for i in server.banned_ids:
        malice_pred[i] = 1

    print(classification_report(malice_label, malice_pred, target_names=['benign', 'malice']))
