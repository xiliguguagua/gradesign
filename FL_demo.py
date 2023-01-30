import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from pyod.models.vae import VAE

N = 100  # user num
T = 1  # total communication round
M = 10  # shuffler num
k = 10  # least num in a shuffler
batch = 32
local_lr = [0.0001, 0.1]
fA_coeff = 0.001
fC_coeff = 1
B_coeff = 0.5
DMS_rho = 1
DMS_h = 0.01



emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = next(iter(example_dataset))
example_element['label'].numpy()