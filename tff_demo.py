import collections
import time

import global_var as gl
from compute_blocks import *
from network import *
from utils import *


n_sample = 300
logdir = 'logdir'
if __name__ == '__main__':
    federated_train_data = []
    for uid in range(gl.args.N):
        train_dataset = gl.train_dataset.skip(n_sample * uid).take(n_sample).cache().repeat(gl.args.epoch).batch(gl.args.batch_size)
        federated_train_data.append(train_dataset)

    if gl.args.task == 'emnist/mnist':
        eval_model = create_E_model(gl.input_shape)
    elif gl.args.task == 'cifar10':
        eval_model = create_C_model(gl.input_shape)
    else:
        eval_model = None
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # metric = tf.keras.metrics.SparseCategoricalCrossentropy()

    server_state = federated_algorithm.initialize()
    start_time = time.time()

    # communication rounds
    for t in range(gl.args.T):

        if t == gl.args.T - 1:
            with tf.profiler.experimental.Profile(logdir):
                result = federated_algorithm.next(
                    server_state, federated_train_data)
        else:
            result = federated_algorithm.next(
                server_state, federated_train_data)

        server_state = result.state
        train_metrics = result.metrics['client_work']['train']
        print(f'Round {t} training loss: {train_metrics["loss"]}, '
              f'time: {(time.time() - start_time) / (t + 1.)} secs')

        model_weights = federated_algorithm.get_model_weights(server_state)
        model_weights.assign_weights_to(eval_model)
        accuracy = keras_evaluate(eval_model, gl.test_dataset, metric)
        print(f'Round {t} validation accuracy: {accuracy * 100.0}')
