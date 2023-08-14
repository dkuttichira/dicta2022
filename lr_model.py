# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:21:03 2022

@author: deept
"""

from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist



seed=21
class_num=10
epochs=10
optimizer='adam'

(x_train,y_train),(x_test,y_test)=mnist.load_data()
y_test_check=y_test



x_train= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist15n_train.npy')
x_test= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist15n_test.npy')
val=np.amax(x_train)
x_train=x_train/val
x_test=x_test/val


y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


#model construction
input_dim = 15 
output_dim = 10 

def classification_model():
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#training
keras_model = classification_model()
keras_model.fit(x_train, y_train, epochs=epochs, verbose=1)

prediction=keras_model.predict(x_test)
prediction_class=keras_model.predict_classes(x_test)

accuracy=accuracy_score(prediction_class, y_test_check)

keras_model.save(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist_model_logistic20exp.h5')