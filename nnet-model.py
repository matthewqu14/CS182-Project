import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

dimensions = [25, 50, 100, 200]
for VECTOR_DIMENSION in [25]:
    depressed_data = np.genfromtxt(f"vectors/all_depressed_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    normal_data = np.genfromtxt(f"vectors/all_normal_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    parameter_space = {
        'solver': ["sgd", "adam"],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'hidden_layer_sizes': [(10, 10), (50, 50), (100, 100), (100,)]
    }

    mlp = MLPClassifier(max_iter=3000,
                        random_state=0)
    # mlp.fit(X_train, y_train)
    #
    # print(f"Accuracy on training set: {mlp.score(X_train, y_train):.5f}")
    # print(f"Accuracy on test set: {mlp.score(X_test, y_test):.5f}")

    clf = GridSearchCV(mlp, parameter_space, cv=5)
    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")
