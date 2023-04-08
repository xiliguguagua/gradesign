import tensorflow as tf
import tensorflow_federated as tff
from network import model_fn

import global_var as gl


@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables


@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)


with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
tf_dataset_type = tff.SequenceType(model.input_spec)
model_weights_type = server_init.type_signature.result


@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    print('in')
    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)

        # Compute the gradient norm and clip
        gradient_norm = tf.linalg.global_norm(grads)
        grads = tf.nest.map_structure(lambda x: x / tf.maximum(1., gradient_norm / gl.args.clip), grads)

        grads_and_vars = zip(grads, client_weights)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)

    return client_weights


@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    if gl.args.task == 'emnist/mnist':
        client_optimizer = tf.keras.optimizers.Adam(learning_rate=gl.args.local_lr, decay=0.0001)
    elif gl.args.task == 'cifar10':
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=gl.args.local_lr, decay=0.0001)
    else:
        return None
    return client_update(model, tf_dataset, server_weights, client_optimizer)


@tf.function
def server_update(model, target_weights):
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, target_weights)
    return model_weights


@tff.tf_computation(model_weights_type)
def server_update_fn(target_weights):
    model = model_fn()
    return server_update(model, target_weights)


federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_weights)

    # Each client computes their updated weights.
    client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client))

    # The server averages these updates.
    mean_client_weights = tff.federated_mean(client_weights)  # --------------------------------------------------------

    # The server updates its model.
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights


# federated_algorithm = tff.templates.IterativeProcess(
#     initialize_fn=initialize_fn,
#     next_fn=next_fn
# )

federated_algorithm = tff.learning.templates.compose_learning_process(
    initial_model_weights_fn,
    distributor,
    client_work,
    aggregator,
    finalizer
)
