#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:24:55 2019

@author: rajat
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def plot_predictions(y_test, y_pred, X_test):
    incorrect_pred_nocell = np.intersect1d(np.where(y_test==0)[0], np.where(y_pred==1)[0])
    incorrect_pred_cell = np.intersect1d(np.where(y_test==1)[0], np.where(y_pred==0)[0])
    correct_pred_cell = np.intersect1d(np.where(y_test==1)[0], np.where(y_pred==1)[0])
    correct_pred_nocell = np.intersect1d(np.where(y_test==0)[0], np.where(y_pred==0)[0])
    plt.figure()
    plt.subplot(221)
    plt.imshow(X_test[incorrect_pred_nocell[-1]], cmap='Greys')
    plt.subplot(222)
    plt.imshow(X_test[incorrect_pred_cell[-1]], cmap='Greys')
    plt.subplot(223)
    plt.imshow(X_test[correct_pred_cell[10]], cmap='Greys')
    plt.subplot(224)
    plt.imshow(X_test[correct_pred_nocell[-1]], cmap='Greys')
    plt.show()

X = np.load('X2.npy')
Y = np.load('Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
del X,Y

plt.figure()
plt.subplot(121)
plt.imshow(X_train[np.where(y_train==1)[0][0]], cmap='Greys')
plt.subplot(122)
plt.imshow(X_train[np.where(y_train==0)[0][-1]], cmap='Greys')
plt.show()

nsamples, nx, ny = X_train.shape
X2_train = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
X2_test = X_test.reshape((nsamples,nx*ny))

# versatile nad requires little tuning 
"""
A random forest is a meta estimator that fits a number of decision tree classifiers on
 various sub-samples of the dataset and uses averaging to improve the predictive accuracy 
 and control over-fitting.
 """
clf_rf = RandomForestClassifier()
clf_rf.fit(X2_train, y_train)
y_pred_rf = clf_rf.predict(X2_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
plot_predictions(y_test, y_pred_rf, X_test)
print("random forest accuracy: ",acc_rf)

# Stohastic gradient, not great accuracy
"""
This estimator implements regularized linear models with stochastic gradient descent (SGD) 
learning: the gradient of the loss is estimated each sample at a time and the model is
 updated along the way with a decreasing strength schedule (aka learning rate). 
"""
clf_sgd = SGDClassifier()
clf_sgd.fit(X2_train, y_train)
y_pred_sgd = clf_sgd.predict(X2_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
plot_predictions(y_test, y_pred_sgd, X_test)
print("stochastic gradient descent accuracy: ",acc_sgd)

# slower than SGD
"""
Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than 
libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale 
better to large numbers of samples.
"""
clf_svm = LinearSVC()
clf_svm.fit(X2_train, y_train)
y_pred_svm = clf_svm.predict(X2_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
plot_predictions(y_test, y_pred_svm, X_test)
print("Linear SVM accuracy: ",acc_svm)

# KNN, slowest, decent accuracy
"""
Classifier implementing the k-nearest neighbors vote.
"""
clf_knn = KNeighborsClassifier()
clf_knn.fit(X2_train, y_train)
y_pred_knn = clf_knn.predict(X2_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
plot_predictions(y_test, y_pred_knn, X_test)
print("nearest neighbors accuracy: ",acc_knn)


# Keras based classifier 
# Reshaping the array to 4-dims so that it can work with the Keras API
def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    return x_train, x_test, input_shape

def createModel(conv_size):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(conv_size, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation=tf.nn.softmax)) # since we have only 2 classes: cell or no-cell
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

"""
Convolution Neural Network with a 2d convolution -> 2d max pooling -> dense layer -> dropout 
-> dense layer (of size 2; dependent on the number of classes)
"""    
x_train, x_test, input_shape = preprocess_data(X_train, X_test)
model = createModel(int(input_shape[0]/1.5))
model.fit(x=x_train,y=y_train, epochs=100)
print("CNN performance")
model.evaluate(x_test, y_test)

# predict individual images
#image_index = 4
#plt.figure()
#plt.imshow(x_test[image_index].reshape(50, 50),cmap='Greys')
#pred = model.predict(x_test[image_index].reshape(1, 50, 50, 1))
#print(pred.argmax())
