import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


def create_ts(start='2000', n=200, freq='M'):
    rng = pd.date_range(start=start, periods=n, freq=freq)
    # cumsum: accumulate the value prior to current index
    ts = pd.Series(np.random.uniform(-18, 18, size=len(rng)), rng).cumsum()
    return ts

ts = create_ts(start='2001', n=222, freq='M')
#print(ts.tail(5))

## left
#plt.figure(figsize=(11, 4))
#plt.subplot(121)
#plt.plot(ts.index, ts)
#plt.plot(ts.index[90:100], ts[90:100], "g-", linewidth=3, label="A training instance")
#plt.title("A time series (generated)", fontsize=14)
##print(ts)
#
## right
#plt.subplot(122)
#plt.title("A training instance", fontsize=14)
#plt.plot(ts.index[90:100], ts[90:100], "b-", markersize=8, label="instance")
#plt.plot(ts.index[91:101], ts[91:101], "bo", markersize=10, label="target", markerfacecolor='red')
#plt.legend(loc="upper left")
#plt.xlabel("Time")
#
#plt.show()


# 222 dataset, 201 for training, 21 for test
# X batches 20 batches of size 10*1, Y has the same shape but one period ahead.
# X batches are lagged by 1 period (take value t-1)
# X batches 1:200, Y batches 2:201

series = np.array(ts)
n_windows = 20
n_input = 1
n_output =1
size_train = 201
n_neuron = 120

# split data
train = series[:size_train]
test = series[size_train:]
print(train.shape, test.shape)

## it will use train data for test array
## that means to test the trained data, the predicted results are much better
#def create_batches(df, windows, input, output):
#    x_data = train[:size_train-1]
#    X_batches = x_data.reshape(-1, windows, input)
#
#    y_data = train[n_output:size_train]
#    y_batches = y_data.reshape(-1, windows, output)
#    return X_batches, y_batches

def create_batches(df, windows, input, output):
    x_data = df[:len(df)-1]
    X_batches = x_data.reshape(-1, windows, input)

    y_data = df[n_output:len(df)]
    y_batches = y_data.reshape(-1, windows, output)
    return X_batches, y_batches

X_batches, y_batches = create_batches(df=train,
                                      windows=n_windows,
                                      input=n_input,
                                      output=n_output)
#print(X_batches.shape, y_batches.shape)

X_test, y_test = create_batches(df=test,
                                windows=n_windows,
                                input=n_input,
                                output=n_output)
#print(X_test.shape, y_test.shape)

# Create RNN

# 1. construct the tensors
X = tf.placeholder(tf.float32, [None, n_windows, n_input])
y = tf.placeholder(tf.float32, [None, n_windows, n_output])

# 2. create the model
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neuron, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

stacked_rnn_output = tf.reshape(rnn_output, [-1, n_neuron])
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])

# 3. Loss and optimization
learning_rate = 0.001
loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

iteration = 1500

with tf.Session() as ss:
    init.run()
    for iters in range(iteration):
        ss.run(training_op, feed_dict={X: X_batches, y: y_batches})
        if iters % 150 == 0:
            mse = loss.eval(feed_dict={X: X_batches,y: y_batches})
            print(iters, "\tMES:", mse)

    y_pred = ss.run(outputs, feed_dict={X: X_test})

plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(y_test)), "bo", markersize=8, label="Actual", color='green')
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=8, label="Forecast", color='red')
plt.legend(loc="lower left")
plt.xlabel("Time")

plt.show()
