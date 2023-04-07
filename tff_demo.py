import argparse
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from pyod.models.vae import VAE

local_lr = [0.0001, 0.1]


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
    parser.add_argument("--epoch", type=int, default=500,
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
    return args


def preprocess(dataset):
    def batch_formate_fn(element):
        return collections.OrderedDict(
            x=element['pixels'],
            y=tf.reshape(element['label'], [-1, 1])
        )

    return dataset.repeat(args.local_epoch).shuffle(args.shuffle_size, seed=1).batch(args.batch_size).map(
        batch_formate_fn).prefetch(args.prefetch_size)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=2, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=10),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
    preprocessed_example_dataset = preprocess(example_dataset)

    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


args = get_args()
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

if __name__ == '__main__':
    sample_clients = emnist_train.client_ids[:args.N]
    federated_train_data = make_federated_data(emnist_train, sample_clients)

    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    train_state = training_process.initialize()

    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    train_metrics = result.metrics
    print('round  1, metrics={}'.format(train_metrics))

    for round_num in range(2, args.T):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, train_metrics))
