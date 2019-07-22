import tensorflow as tf
import numpy as np

# TODO: numpy random
x_input = np.random.sample((1, 3))
print("x input is: %r" % (x_input))


x = tf.placeholder(tf.float32, shape=[1,3], name="x")

# create dataset
dataset = tf.data.Dataset.from_tensor_slices(x)

# create the pipeline
iterator = dataset.make_initializable_iterator()
next = iterator.get_next()

with tf.Session() as ss:
    ss.run(iterator.initializer, feed_dict={x: x_input})
    # it can be got only once
    print("get the random num {}".format(ss.run(next)))

