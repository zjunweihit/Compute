import tensorflow as tf

hello = tf.constant('Hello Tensorflow')
ss = tf.Session()

print(ss.run(hello))

ss.close()
