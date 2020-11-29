import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dimensions = [25, 50, 100, 200]
for VECTOR_DIMENSION in [25]:
    depressed_data = np.genfromtxt(f"vectors/all_depressed_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    normal_data = np.genfromtxt(f"vectors/all_normal_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)

    # best performance ~0.885
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)

    print(f"Accuracy on training set: {clf.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")
