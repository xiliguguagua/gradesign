import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend

def load_data(args):
    ds, info = tfds.load(args.task, data_dir='../dataset/', batch_size=args.batch_size,
                         with_info=True)
    train_dataset, test_dataset = ds['train'], ds['test']
    input_shape = info.features['image'].shape
    num_classes = info.features['label'].num_classes
    return train_dataset, test_dataset, input_shape, num_classes

def flatten(gradients):
    shapes = [x.shape for x in gradients]
    return tf.concat([backend.flatten(x) for x in gradients], axis=0), shapes