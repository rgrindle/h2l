import keras
import matplotlib.pyplot as plt
import numpy as np

import os
print(os.getcwd())

# load json and create model
json_file = open('model_digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_digit.h5")
print("Loaded model from disk")

# load mnist dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() # everytime loading data won't be so easy :)

X_test_original = X_test

img_rows = 28
img_cols = 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)

#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

## Evaluating a MNIST number
# index = 1
# prediction = loaded_model.predict(np.array([X_test[index]]))
#
# print(prediction)
# print('sum', sum(prediction[0]))
# print(np.argmax(prediction))
#
# fig = plt.figure()
#
# # plt.subplot(3,3,i+1)
# plt.tight_layout()
# plt.imshow(X_test_original[index], cmap='gray', interpolation='none')
# # plt.title("Digit: {}".format(y_train[i]))
# # plt.xticks([])
# # plt.yticks([])
#
# plt.show()

# Loading our own image
img = keras.preprocessing.image.load_img('test_white.jpg')
array_img = keras.preprocessing.image.img_to_array(img)
print(array_img)
print(array_img.shape)

array_img2 = np.mean(array_img[:,:])

array_img2 /= 255

print(array_img2)
print(array_img2.shape)

index = 1
prediction = loaded_model.predict(np.array([X_test[index]]))

print(prediction)
print('sum', sum(prediction[0]))
print(np.argmax(prediction))

fig = plt.figure()

# plt.subplot(3,3,i+1)
plt.tight_layout()
plt.imshow(img, cmap='gray', interpolation='none')
# plt.title("Digit: {}".format(y_train[i]))
# plt.xticks([])
# plt.yticks([])

plt.show()