import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

dimensions = [25, 50, 100, 200]
for VECTOR_DIMENSION in [25]:
    # retrieve data
    depressed_data = np.genfromtxt(f"vectors/all_depressed_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    normal_data = np.genfromtxt(f"vectors/all_normal_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

    # scale feature vectors to 0 mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:, :-1])

    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(data[:, -1:]), random_state=0)

    # best performance ~0.81610, 0.84371, 0.85488; weights="distance"; n_neighbors=18, 10, 12
    clf = KNeighborsClassifier(n_neighbors=51, weights="uniform")
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    print(f"Binary cross-entropy loss: {log_loss(y_test, y_pred):.5f}")
    print(f"Accuracy on training set: {clf.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")

    # Grid search with 5-fold CV to find optimal hyperparameters
    # parameter_space = {
    #     "weights": ["uniform", "distance"],
    #     "n_neighbors": [_ for _ in range(1, 211, 10)],
    # }
    # clf = GridSearchCV(KNeighborsClassifier(), parameter_space, scoring="accuracy", cv=5)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)
    # print(f"Best cross-validation score: {clf.best_score_:.5f}")
    # print(f"Accuracy on test set: {clf.score(X_train, y_train):.5f}")
    # print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")
