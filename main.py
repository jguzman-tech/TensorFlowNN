# This Python file uses the following encoding: iso-8859-1

import argparse
import pdb # use pdb.set_trace() to set a "break point" when debugging
import os, sys
import numpy as np
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path
import copy
import warnings
import statistics
from matplotlib.pyplot import figure
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# warnings.simplefilter('error') # treat warnings as errors
figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

## Load the spam data set and Scale the input matrix
def Parse(fname):
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    np.random.shuffle(temp_ar)
    return temp_ar

if __name__ == "__main__":
    temp_ar = Parse("spam.data")
    X = temp_ar[:, 0:-1] # m x n
    X = X.astype(float)
    y = np.array([temp_ar[:, -1]]).T 
    y = y.astype(int)
    num_row = X.shape[0]

## Divide the data into 80% train, 20% test observations (out of all observations in the whole data set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Next divide the train data into 60% subtrain, 40% validation.
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(X, y, test_size=0.4)

## Define three different neural networks
# for first model
# setup layers
model_one = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain.shape[1]),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(1)
])
# compiler model
model_one.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])

# for second model
# setup layers
model_two = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain.shape[1]),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(1)
])
# compiler model
model_two.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])

# for third model
# setup layers
model_three = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain.shape[1]),
    keras.layers.Dense(1000, activation='sigmoid'),
    keras.layers.Dense(1)
])
# compiler model
model_three.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])


# feed the model
model_one.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=10)






























