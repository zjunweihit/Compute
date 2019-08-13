from keras.utils import to_categorical
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

img_rows, img_cols = 28, 28

# load test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_test = x_test.astype('float32')
# scale values to a range of 0 to 1 by dividing 255
x_test /= 255.0

# one hot
y_test = to_categorical(y_test, 10)

print("Loading pre-trained model ...")
model = load_model("keras_cnn.model")

# prediction and show the images
test_total = 9
plt_row, plt_col = 3, 3

# pick the numbers randomly
index = np.random.randint(len(x_test), size=test_total)

# 0 ~ (test_total-1)
for i in range(test_total):
    image = x_test[index[i]].reshape(1, img_rows, img_cols, 1)
    pred_arr = model.predict(image)
    pred_num = pred_arr.argmax()

    image = image.reshape(img_rows, img_cols)
    # plot from index 1
    plt.subplot(plt_row, plt_col, i+1)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title("prediction({})".format(index[i]))
    plt.text(2, 3, "{}".format(pred_num), size=15, rotation=30., ha="center", va="center",
             bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8)))

plt.tight_layout()
plt.show()
