import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

dimensions = [25, 50, 100, 200]
for VECTOR_DIMENSION in dimensions:
    # retrieve data
    depressed_data = np.genfromtxt(f"vectors/all_depressed_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    normal_data = np.genfromtxt(f"vectors/all_normal_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=1)

    # single decision tree

    # tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=0)
    # tree.fit(X_train, y_train)
    # y_pred = tree.predict_proba(X_test)
    # print(f"Binary cross-entropy loss: {log_loss(y_test, y_pred):.5f}")
    # print(f"Accuracy on training set for decision tree: {tree.score(X_train, y_train):.5f}")
    # print(f"Accuracy on test set for decision: {tree.score(X_test, y_test):.5f}")

    # Grid search with 5-fold CV to find optimal hyperparameters
    parameter_space_tree = {
        "max_depth": [_ for _ in range(1, 21)],
    }
    clf = GridSearchCV(DecisionTreeClassifier(random_state=0), parameter_space_tree, scoring="accuracy", cv=5)
    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    print(f"Best cross-validation score: {clf.best_score_:.5f}")
    print(f"Accuracy on training set: {clf.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")

    # random forest
    # best performance 0.81571, 0.84019, 0.85409, 0.85703; max_depth=15

    # forest = RandomForestClassifier(criterion="entropy", n_estimators=128, max_depth=10, random_state=0)
    # forest.fit(X_train, y_train)
    # y_pred = forest.predict_proba(X_test)
    # print(f"Binary cross-entropy loss: {log_loss(y_test, y_pred):.5f}")
    # print(f"Accuracy on training set for random forest: {forest.score(X_train, y_train):.5f}")
    # print(f"Accuracy on test set for random forest: {forest.score(X_test, y_test):.5f}")

    # Grid search with 5-fold CV to find optimal hyperparameters
    parameter_space = {
        "max_depth": [_ for _ in range(1, 21)],
    }
    clf = GridSearchCV(RandomForestClassifier(n_estimators=128, random_state=0), parameter_space, cv=5)
    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    print(f"Best cross-validation score: {clf.best_score_:.5f}")
    print(f"Accuracy on training set: {clf.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set: {clf.score(X_test, y_test):.5f}")

# plot decision tree visualization
# plt.figure(figsize=[19.2, 14.4])
# plot_tree(tree, max_depth=3, class_names=["normal", "depressed"], filled=True, fontsize=12)
# plt.show()
