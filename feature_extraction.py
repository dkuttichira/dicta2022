# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:59:22 2022

@author: deept
"""
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import scipy.io


layer_number=5

# Loading and preprocessing dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= (x_train.astype('float32'))/255
x_test= (x_test.astype('float32'))/255
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
model=load_model(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist_15n.h5')


for i in range(len(model.layers)):
 	layer = model.layers[i]

 	print(i, layer.name, layer.output.shape)

 
new_model = Model(inputs=model.inputs, outputs=model.layers[layer_number].output)

test_features=new_model.predict(x_test)
train_features=new_model.predict(x_train)

np.save(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist15n_testflatten.npy',test_features)
np.save(r'C:\Users\deept\.spyder-py3\CNN_feature\MNIST_deep\mnist15n_trainflatten.npy',train_features)

