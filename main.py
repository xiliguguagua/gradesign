import argparse

from utils import *
from user import User


def get_args():
    parser = argparse.ArgumentParser(description="train MSFL")
    parser.add_argument("--N", type=int, default=100,
                        help="user num")
    parser.add_argument("--M", type=int, default=10,
                        help="shuffler num")
    parser.add_argument("--T", type=int, default=150,
                        help="total communication round")
    parser.add_argument("--k", type=int, default=10,
                        help="least user num in a shuffler")
    parser.add_argument("--local_epoch", type=int, default=2,
                        help="epoch for each user in one communication turn")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="batch size")
    parser.add_argument("--shuffle_size", type=int, default=100,
                        help="shuffle size to random choose sample for user in one step")
    parser.add_argument("--prefetch_size", type=int, default=10,
                        help="prefetch size")
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_dataset, test_dataset = load_data()
    users = []
    for i in range(args.N):
        users.append(User(args))