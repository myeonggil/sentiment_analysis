import mglearn
import graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#print(mglearn.plots.plot_logistic_regression_graph())
#print(mglearn.plots.plot_single_hidden_layer_graph())
#print(mglearn.plots.plot_two_hidden_layer_graph())

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(x_train, y_train)

#print('Accuracy on the training subset: {:.3f}'.format(mlp.score(x_train, y_train)))
#print('Accuracy on the testing subset: {:.3f}'.format(mlp.score(x_test, y_test)))

#print('The maximum per each feature:\m{}'.format(cancer.data.max(axis=0)))

scaler = StandardScaler()
x_train_scaler = scaler.fit(x_train).transform(x_train)
x_test_scaled = scaler.fit(x_test).transform(x_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(x_train_scaler, y_train)

#print('Accuracy on the training subset: {:.3f}'.format(mlp.score(x_train_scaler, y_train)))
#print('Accuracy on the testing subset: {:.3f}'.format(mlp.score(x_test_scaled, y_test)))

#print(mlp)

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(x_train_scaler, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(x_train_scaler, y_train)))
print('Accuracy on the testing subset: {:.3f}'.format(mlp.score(x_test_scaled, y_test)))

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='None', cmap='GnBu')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()
plt.show()