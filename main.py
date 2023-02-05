import argparse
import random

import numpy as np

from utils import *
from user import User
from server import Server


def get_args():
    parser = argparse.ArgumentParser(description="MSFL")
    parser.add_argument("--task", type=str, default='emnist')
    parser.add_argument("--N", type=int, default=100,
                        help="user num")
    parser.add_argument("--Na", type=int, default=5,
                        help="attacker num")
    parser.add_argument("--M", type=int, default=10,
                        help="shuffler num")
    parser.add_argument("--T", type=int, default=150,
                        help="total communication round")
    parser.add_argument("--k", type=int, default=10,
                        help="least user num in a shuffler")
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--global_lr", type=float, default=0.0001)
    parser.add_argument("--max_it", type=int, default=2,
                        help="max epoch in local train")
    parser.add_argument("--min_it", type=int, default=1,
                        help="min epoch in local train")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="batch size")
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
    parser.add_argument("--AAE_alpha", type=float, default=2,
                        help="alpha in AAE")
    parser.add_argument("--od_method", type=str, default='all',
                        help="to od among each shuffler or all users")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_dataset, test_dataset, input_shape = load_data()
    server = Server(args, input_shape)
    users = []

    # generate malice user id
    malice_idset = set()
    malice_label = [0] * args.N
    for i in range(args.Na):
        m_id = random.randint(0, args.N)
        while m_id in malice_idset:
            m_id = random.randint(0, args.N)
        malice_idset.add(m_id)
        malice_label[m_id] = 1

    # split dataset to all users
    ptr = 0
    for i in range(args.N):
        n = random.randint(200, 400)
        users.append(User(i, args, malice_label[i],
                          train_dataset[ptr:ptr+n], test_dataset[ptr:ptr+n],
                          input_shape, n))
        ptr += n

    # each communication turn
    for t in range(args.T):
        # sync global model
        for user in users:
            user.update_model(server.global_model)
        # local train
            user.local_train()
