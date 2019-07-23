#
# Scalars: Show different useful information during the model training
# Graphs: Show the model
# Distribution: Display the distribution of the weight
# Histogram: Display weights with a histogram
# Projector: Show Principal component analysis and T-SNE algorithm.
#            The technique uses for dimensionality reduction
#
# To call tensorboard with the result log
#   $ tensorboard --logdir=./train/dnnreg
#
# Open the link in browser

import tensorflow as tf
import numpy as np

X_train = (np.random.sample((10000,5)))
y_train =  (np.random.sample((10000,1)))
X_train.shape

feature_columns = [
      tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]
DNN_reg = tf.estimator.DNNRegressor(feature_columns=feature_columns,
     model_dir='train/dnnreg', # Indicate where to store the log file
     hidden_units=[500, 300],
     optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001
      )
)

train_input = tf.estimator.inputs.numpy_input_fn(
     x={"x": X_train},
     y=y_train, shuffle=False,num_epochs=None)
DNN_reg.train(train_input,steps=3000)
