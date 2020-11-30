#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:18:45 2020

@author: alfiantjandra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import timeit





# #Vector dimension: 25,50,100,200
# dimensions = [25,50,100,200]

# #Vector Type: "lemmatized_" , "stemmed_", or ""  ***remember the _ ***
# vector_types = ["lemmatized_"]




# for dimension in dimensions:
#     for vector_type in vector_types:
        
#         #Folder name
#         path_depressed = vector_type + "vectors/all_depressed_vectors_"+ str(dimension)+".csv"
#         path_normal = vector_type + "vectors/all_normal_vectors_"+ str(dimension)+".csv"
        
        
        
#         depressed_data = np.genfromtxt(path_depressed, delimiter=",")
#         normal_data = np.genfromtxt(path_normal, delimiter=",")
#         data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)
        
#         X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)
        
#         X_train = preprocessing.scale(X_train)
#         X_test = preprocessing.scale(X_test)
        
        
        # 'Finding optimal C_values'
        # parameter_space = {
        #         "C": [10**i for i in range(-10,10)],
        #     }
        # clf = GridSearchCV(LogisticRegression(), parameter_space,cv=5)
        # clf.fit(X_train, y_train)
        # print('Best parameters found for '+vector_type+"and dimension " +str(dimension)+':\n', clf.best_params_)
        # print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")
        # print(f"Accuracy on training set: {clf.score(X_train, y_train):.5f}")
        
        # ''' Optimal C_values '''
        # Best parameters found for lemmatized_and dimension 25:
        #   {'C': 1}
        # Accuracy on test set: 0.75264
        # Accuracy on training set: 0.75323
        # Best parameters found for lemmatized_and dimension 50:
        #   {'C': 0.1}
        # Accuracy on test set: 0.79514
        # Accuracy on training set: 0.79495
        # Best parameters found for lemmatized_and dimension 100:
        #   {'C': 0.1}
        # Accuracy on test set: 0.84587
        # Accuracy on training set: 0.84038
        # Best parameters found for lemmatized_and dimension 200:
        #   {'C': 0.01}
        # Accuracy on test set: 0.85919
        # Accuracy on training set: 0.85847



#Folder name
path_depressed = "lemmatized_vectors/all_depressed_vectors_200.csv"
path_normal = "lemmatized_vectors/all_normal_vectors_200.csv"



depressed_data = np.genfromtxt(path_depressed, delimiter=",")
normal_data = np.genfromtxt(path_normal, delimiter=",")
data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

     
training_accuracy = []
test_accuracy = []

C_value = [0.01]

for c in C_value:
    clf = LogisticRegression(C=c)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
    
print(max(test_accuracy))
# print(np.argmax(test_accuracy)+)
# plt.plot(C_value, training_accuracy, label="training accuracy")
# plt.plot(C_value, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("log_2_C_value")
# plt.show()


