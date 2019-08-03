import tensorflow as tf
import numpy as np

# todo: Use keras.layers instead, but the parameters are changed as well

# show tensorflow info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

def cnn_model_fn(features, labels, mode):
    # one element of [28 * 28], as many as the features could have
    # (55000, 784) => (55000, 28, 28, 1)
    # -1, as many as possible
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # 1st convolutional layer
    # output (batch_size, 28, 28, 32) 32 filters to filter each image 14 times
    # If the number of filers is 14, the process will be error.
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    print(conv1.shape)

    # 1st pooling layer
    # output (batch_size, 14, 14, 32)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)
    print(pool1.shape)

    # 2nd convolutional layer
    # output (batch_size, 14, 14, 36)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=36,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    print(conv2.shape)

    # 2nd pool layer
    # output (batch_size, 7, 7, 36)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)
    print(pool2)

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 36])
    dense = tf.layers.dense(inputs=pool2_flat, units=7*7*36, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.3, training= (mode == tf.estimator.ModeKeys.TRAIN))

    # Logit layer
    logit = tf.layers.dense(inputs=dropout, units=10)

    # prediction for "PREDICT" and "EVAL" mode
    predictions = {
        "classes": tf.argmax(input=logit, axis=1),
        "probabilities": tf.nn.softmax(logit, name="softmax_tensor")
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logit)

    # Train for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation for EVALUATION mode
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Create Estimator
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="train")

# Set up logging for perdictions
#   store the value every 50 iteratioins
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


def get_input_fn(dataset, n_epochs, b_size, shuffle):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": dataset.images},
        y=dataset.labels.astype(np.int32),
        num_epochs=n_epochs,
        batch_size=b_size,
        shuffle=shuffle
    )


# Train the model
mnist_classifier.train(input_fn=get_input_fn(mnist.train,
                                             None,
                                             128,
                                             True),
                       steps=16000,
                       hooks=[logging_hook])

# Evaluate the model and print results
eval_results = mnist_classifier.evaluate(input_fn=get_input_fn(mnist.test,
                                                               1,
                                                               128,
                                                               False))
print(eval_results)
