#
# Load the result:
#
#   tensorboard --logdir=train/mul
#
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.placeholder(tf.float32, name = "a")
    b = tf.placeholder(tf.float32, name = "b")
    multiply = tf.multiply(a, b, name = "multiply")

    with tf.Session() as ss:
        res = ss.run(multiply, feed_dict={a:[1, 2, 3], b:[4, 5, 6]})
        print(res)

tf.summary.FileWriter("train/mul", g).close()