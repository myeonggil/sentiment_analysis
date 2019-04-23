import mglearn
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

#print('Accuracy on the training subset: {:.3f}'.format(tree.score(x_train, y_train)))
#print('Accuracy on the test subset: {:.3f}'.format(tree.score(x_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(x_train, y_train)

#print('Accuracy on the training subset: {:.3f}'.format(tree.score(x_train, y_train)))
#print('Accuracy on the test subset: {:.3f}'.format(tree.score(x_test, y_test)))

export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

#print('Feature importance: {}'.format(tree.feature_importances_))
n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()