import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# we can use keras mnist dataset as ndarray for
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# scale values to a range of 0 to 1 by dividing 255
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to become one hot data for labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
x, y_acc, y_loss = [], [], []


def get_epoch_data(epoch, logs):
    x.append(epoch + 1) # epoch starts from 0
    y_acc.append(logs.get('accuracy'))
    y_loss.append(logs.get('loss'))

    ax = axes[0]
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.plot(x, y_acc, 'bo-')

    ax = axes[1]
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.plot(x, y_loss, 'r^-')

    plt.pause(1)

PlotInEachEpochCallback = keras.callbacks.LambdaCallback(on_epoch_end=get_epoch_data)

train_history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[PlotInEachEpochCallback],
                          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.show()
