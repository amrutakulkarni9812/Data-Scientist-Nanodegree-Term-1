# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:49:20 2019

@author: amruta.kulkarni
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# X has shape (num_rows, num_cols), where the training data are stored
# as row vectors
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)

# y must have an output vector for each input vector
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
y = np_utils.to_categorical(y)

print(X)
print(y)

 # Create the Sequential model
model = Sequential()
 
# 1st Layer - Add an input layer of 32 nodes with the same input shape as
# the training samples in X
p = model.add(Dense(32, input_dim = X.shape[1])) 
print(p) 

# Add a softmax activation layer
model.add(Activation('softmax'))

# 2nd Layer - Add a fully connected output layer
model.add(Dense(2))

# Add a sigmoid activation layer
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer= "adam",  metrics = ["accuracy"])

model.summary()

model.fit(X, y, epochs= 1000, verbose = 0)

model.evaluate(X, y)