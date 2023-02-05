def load_data():
    return None, None, None

def flatten(gradients):
    shapes = [x.shape for x in gradients]
    return tf.concat([backend.flatten(x) for x in gradients], axis=0), shapes