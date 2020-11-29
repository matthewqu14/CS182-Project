import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

dimensions = [25, 50, 100, 200]
for VECTOR_DIMENSION in [25]:
    depressed_data = np.genfromtxt(f"vectors/all_depressed_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    normal_data = np.genfromtxt(f"vectors/all_normal_vectors_{VECTOR_DIMENSION}.csv", delimiter=",")
    data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)

    # single decision tree, best performance ~0.865
    tree = DecisionTreeClassifier(max_depth=7, random_state=0)
    tree.fit(X_train, y_train)
    print(f"Accuracy on training set for single decision tree: {tree.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set for single decision tree: {tree.score(X_test, y_test):.5f}")

    # random forest, best performance ~0.895
    forest = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)
    forest.fit(X_train, y_train)
    print(f"Accuracy on training set for random forest: {forest.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set for random forest: {forest.score(X_test, y_test):.5f}")

    # gradient-boosted regression tree, best performance ~0.897
    gbrt = GradientBoostingClassifier(random_state=0, n_estimators=160, learning_rate=0.1, max_depth=6)
    gbrt.fit(X_train, y_train)
    print(f"Accuracy on training set for gradient-boosted tree: {gbrt.score(X_train, y_train):.5f}")
    print(f"Accuracy on test set for gradient-boosted tree: {gbrt.score(X_test, y_test):.5f}")

    # print(tree.feature_importances_)

# for decision tree visualization, not working yet
#     export_graphviz(tree, out_file="tree.dot", class_names=["depressed", "normal"])
#
# with open("tree.dot") as f:
#     graph = f.read()
#
# pic = graphviz.Source(graph)
# pic.view()
