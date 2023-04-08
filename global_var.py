import argparse
from utils import *

parser = argparse.ArgumentParser(description="MSFL")
parser.add_argument("--task", type=str, default='cifar10')
parser.add_argument("--N", type=int, default=5,
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
parser.add_argument("--epoch", type=int, default=5,
                    help="max epoch in local train")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument("--clip", type=float, default=15,
                    help="weight norm2 clip")
parser.add_argument("--epsilon", type=float, default=1.,
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

train_dataset, test_dataset, input_shape = load_data(args)
element_spec = train_dataset.batch(args.batch_size).element_spec
