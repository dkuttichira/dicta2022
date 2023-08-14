# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:42:48 2021

@author: deept
"""


# deeper cnn model for mnist

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt


seed=21
class_num=10
epochs=10
optimizer='adam'
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test_check=y_test


x_train= (x_train.astype('float32'))/255
x_test= (x_test.astype('float32'))/255

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    


model = define_model()


np.random.seed(seed)
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=32)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


prediction=model.predict(x_test)
test_class= prediction.argmax(axis=-1)

accuracy=accuracy_score(test_class, y_test_check)

model.save(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist_15new.h5')