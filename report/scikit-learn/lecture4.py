import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#raw_data = pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
#print(raw_data.tail(10))

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(x_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(x_test, y_test)))

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='accuracy of the training')
plt.plot(neighbors_settings, test_accuracy, label='accuracy of the test')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.grid()
plt.show()