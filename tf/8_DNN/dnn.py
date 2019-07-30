# reference: https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940

import tensorflow as tf
import numpy as np

# show tensorflow info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# -----------------------------------------------------------------------
## version 0.20, deprecates fetch_mldata, 0.22 will remove it completely
## both fetch_mldata and fetch_openml are going to download the mnist data(.mat for mldata and 784 for openml)
## it doesn't work for my network to downlad them.

#from sklearn.datasets import fetch_mldata
#from sklearn.datasets import fetch_openml

#mnist = fetch_mldata('MNIST original', data_home='MNIST/mldata/dataset')
#mnist = fetch_openml('mnist_784')

## it looks to read the raw data from the dataset, don't know how to use it by now
#from mlxtend.data import loadlocal_mnist

#x, y = loadlocal_mnist(images_path='data/train-images-idx3-ubyte',
#                       labels_path='data/train-labels-idx1-ubyte')
#print(x)
#print(y)
# -----------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data

# one_hot:
#   dtype is uint8
#   use 0, 1 to indicate the number
#       [ 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] is label '7'
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/")

#print(mnist)

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="output"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)
classifier.train(input_fn=train_input_fn, steps=10000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

accuracy = classifier.evaluate(input_fn=test_input_fn)["accuracy"]