from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff
from network import model_fn

import global_var as gl


@tff.tf_computation()
def initial_model_weights_fn():
  return tff.learning.models.ModelWeights.from_model(model_fn())

model_weights_type = initial_model_weights_fn.type_signature.result


@tf.function
def client_update(model: tff.learning.models.VariableModel,
                  dataset: tf.data.Dataset,
                  server_weights: tff.learning.models.ModelWeights,
                  client_optimizer: tf.keras.optimizers.Optimizer):
    # Initialize the client model with the current server weights.
    client_weights = tff.learning.models.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    # Keep track of the number of examples as well.
    num_examples = 300.0
    print('train')
    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights.trainable)

        # Compute the gradient norm and clip
        gradient_norm = tf.linalg.global_norm(grads)
        grads = tf.nest.map_structure(lambda x: x / tf.maximum(1., gradient_norm / gl.args.clip), grads)


        grads_and_vars = zip(grads, client_weights.trainable)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)

    # Compute the difference between the server weights and the client weights
    client_update = client_weights.trainable

    return tff.learning.templates.ClientResult(
        update=client_update, update_weight=num_examples)


def build_gradient_clipping_client_work(
        model_fn: Callable[[], tff.learning.models.VariableModel],
        optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
) -> tff.learning.templates.ClientWorkProcess:
    with tf.Graph().as_default():
        # Wrap model construction in a graph to avoid polluting the global context
        # with variables created for this model.
        model = model_fn()
    data_type = tff.SequenceType(model.input_spec)
    model_weights_type = tff.learning.models.weights_type_from_model(model)

    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value((), tff.SERVER)

    @tff.tf_computation(model_weights_type, data_type)
    def client_update_computation(model_weights, dataset):
        model = model_fn()
        optimizer = optimizer_fn()
        return client_update(model, dataset, model_weights, optimizer)

    @tff.federated_computation(
        initialize_fn.type_signature.result,
        tff.type_at_clients(model_weights_type),
        tff.type_at_clients(data_type)
    )
    def next_fn(state, model_weights, client_dataset):
        client_result = tff.federated_map(
            client_update_computation, (model_weights, client_dataset))
        # Return empty measurements, though a more complete algorithm might
        # measure something here.
        measurements = tff.federated_value((), tff.SERVER)
        return tff.templates.MeasuredProcessOutput(state, client_result, measurements)

    return tff.learning.templates.ClientWorkProcess(initialize_fn, next_fn)


@tf.function
def server_update(model, target_weights):
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, target_weights.trainable)
    return model_weights


@tff.tf_computation(model_weights_type)
def server_update_fn(target_weights):
    model = model_fn()
    return server_update(model, target_weights)


distributor = tff.learning.templates.build_broadcast_process(model_weights_type)

if gl.args.task == 'emnist/mnist':
    client_optimizer_fn = lambda: tf.keras.optimizers.Adam(learning_rate=gl.args.local_lr)
elif gl.args.task == 'cifar10':
    client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=gl.args.local_lr)
server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)

client_work = build_gradient_clipping_client_work(model_fn, client_optimizer_fn)
aggregator_factory = tff.aggregators.MeanFactory()
aggregator = aggregator_factory.create(model_weights_type.trainable, tff.TensorType(tf.float32))
finalizer = tff.learning.templates.build_apply_optimizer_finalizer(server_optimizer_fn, model_weights_type)

federated_algorithm = tff.learning.templates.compose_learning_process(
    initial_model_weights_fn,
    distributor,
    client_work,
    aggregator,
    finalizer
)
