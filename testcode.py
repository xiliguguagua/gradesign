import tensorflow as tf
import tensorflow_datasets as tfds
from network import Cifar10Net
from keras import backend as K
print(K.image_data_format())
ds, info = tfds.load("cifar10", data_dir='../dataset/', batch_size=8,
                     shuffle_files=False, download=False, as_supervised=True, with_info=True)

input_shape = info.features.shape['image']
net = Cifar10Net(input_shape)

train_dataset, test_dataset = ds['train'], ds['test']
print(len(train_dataset))

for epoch in range(200):

    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        with tf.GradientTape(persistent=True) as Tape:
            outputs = net(inputs, training=True)
            print(outputs.shape)
            print(outputs)
            input()