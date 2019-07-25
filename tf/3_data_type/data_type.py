import tensorflow as tf
import numpy as np

ss = tf.Session()

# =================== Constant ===================
#
# constant
# D0        tf.constant(1, tf.int16)
# D1        tf.constant([1, 2, 3], tf.int16)
# D2        tf.constant([ [1, 2], [3, 4] ], tf.int16)
# D3        tf.constant([ [[1], [2], [3]], [[4], [5], [6]] ], tf.int16)

c_d1 = tf.constant(1, tf.int16)
# shape (3,) only 3 elements in a row 1x3, like a list
# output:
#   Tensor("Const_1:0", shape=(3,), dtype=int16)
#   [1 2 3]
c_d1_2 = tf.constant([1, 2, 3], tf.int16)
c_d2 = tf.constant([ [1, 2], [3, 4] ], tf.int16)
c_d3 = tf.constant([ [[1], [2], [3]], [[4], [5], [6]] ], tf.int16)

r1 = ss.run(c_d1)
r1_2 = ss.run(c_d1_2)
r2 = ss.run(c_d2)
r3 = ss.run(c_d3)

print("================ Constant ================")
print(c_d1)
print(r1)
print(c_d1_2)
print(r1_2)
print(c_d2)
print(r2)
print(c_d3)
print(r3)

# =================== Variables ===================
#
# After the initialization it CANNOT be used direct, until ss.run(tf.global_variables_initializer()) is called

# initialize by random value
var_init_random = tf.get_variable("var_random", [2, 2], dtype=tf.int32)

# initialize by "0"
var_init_zero = tf.get_variable("var_zero", [1, 2], dtype=tf.int32,  initializer=tf.zeros_initializer)

# initialize by const tensor
const_init = tf.constant([ [1, 2, 3], [4, 5, 6] ], tf.int32)
var_init_const = tf.get_variable("var_init_2", dtype=tf.int32,  initializer=const_init)

# x*y + x^2 + y + c
x = tf.get_variable("x", dtype=tf.int32,  initializer=tf.constant([2]))
y = tf.get_variable("y", dtype=tf.int32,  initializer=tf.constant([3]))
c = tf.constant([5], name =	"constant")
square = tf.constant([2], name ="square")
f = tf.multiply(x, y) + tf.pow(x, square) + y + c

# initialize all variables
ss.run(tf.global_variables_initializer())

r_var_init_random = ss.run(var_init_random)
r_var_init_zero = ss.run(var_init_zero)
r_var_init_const = ss.run(var_init_const)
r_f = ss.run(f)

print("================ Variable ================")

print(r_var_init_random)
print(var_init_random)
print(r_var_init_zero)
print(var_init_zero)
print(r_var_init_const)
print(var_init_const)
print(r_f)
print(f)

# =================== Placeholder ===================
#
# It's used to initialize the data to flow inside the tensors, by "feed_dict"

pl_data = tf.placeholder(tf.float32, name = "pl_data")
power_a = tf.pow(pl_data, 2)

# numpy.random.rand(d0, d1, …, dn) 	Random values in a given shape.
input_data = np.random.rand(1, 10)
r_pow = ss.run(power_a, feed_dict={pl_data: input_data})

print("================ Placeholder ================")
print(r_pow)
print(input_data)

# close the session if created explicitly
# Create a session by block
#   with tf.Session() as sess:
ss.close()