import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

depressed_data = np.genfromtxt("vectors/all_depressed_vectors_25.csv", delimiter=",")
normal_data = np.genfromtxt("vectors/all_normal_vectors_25.csv", delimiter=",")
data = np.concatenate((depressed_data[1:, 1:], normal_data[1:, 1:]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], np.ravel(data[:, -1:]), random_state=0)

training_accuracy = []
test_accuracy = []
neighbors = range(1, 21)

for n_neighbors in neighbors:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

print(max(test_accuracy))
print(np.argmax(test_accuracy) + 1)
plt.plot(neighbors, training_accuracy, label="training accuracy")
plt.plot(neighbors, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.show()
