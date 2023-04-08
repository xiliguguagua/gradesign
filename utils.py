import tensorflow as tf
import tensorflow_datasets as tfds


# from keras import backend


def load_data(args):
    ds, info = tfds.load(args.task, data_dir='../dataset/', as_supervised=True, with_info=True)
    train_dataset, test_dataset = ds['train'], ds['test']
    input_shape = info.features['image'].shape
    # num_classes = info.features['label'].num_classes
    return train_dataset, test_dataset, input_shape


def keras_evaluate(model, test_data, metric):
    metric.reset_states()
    for batch in test_data:
        preds = model(batch['x'], training=False)
        metric.update_state(y_true=batch['y'], y_pred=preds)
    return metric.result()


def CELoss(y_true, y_pred):
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(vector_loss)


# def flatten(weights):
#     shapes = [x.shape for x in weights]
#     return tf.concat([backend.flatten(x) for x in weights], axis=0), shapes


def reconstruct(flat_w, shapes):
    weights = []
    ptr = 0
    for i in range(len(shapes)):
        w_len = tf.math.reduce_prod(shapes[i])
        weights.append(tf.reshape(flat_w[ptr:ptr + w_len], shapes[i]))
        ptr += w_len
    return weights


def merge(trainable, nontrainable, flags):
    new_weights = []
    t_ptr = 0
    nont_ptr = 0
    flag_ptr = 0
    while t_ptr < len(trainable) and nont_ptr < len(nontrainable):
        if flags[flag_ptr] == 1:
            new_weights.append(trainable[t_ptr])
            t_ptr += 1
        else:
            new_weights.append(nontrainable[nont_ptr])
            nont_ptr += 1
        flag_ptr += 1

    if t_ptr < len(trainable):
        new_weights += trainable[t_ptr:]

    if nont_ptr < len(nontrainable):
        new_weights += nontrainable[nont_ptr:]

    return new_weights


def shuffle(weights):
    #  self.m_weights = self.m_weights.shuffle()
    pass


def rebuild(weights):
    #  self.m_weights = self.m_weights.rebuild()
    pass
