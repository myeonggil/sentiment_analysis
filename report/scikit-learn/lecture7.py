from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

#print('Accuracy on the training set: {:.3f}'.format(log_reg.score(x_train, y_train)))
#print('Accuracy on the training set: {:.3f}'.format(log_reg.score(x_test, y_test)))

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(x_train, y_train)

#print('Accuracy on the training set: {:.3f}'.format(log_reg100.score(x_train, y_train)))
#print('Accuracy on the training set: {:.3f}'.format(log_reg100.score(x_test, y_test)))

log_reg001 = LogisticRegression(C=0.01)
log_reg001.fit(x_train, y_train)
print('Accuracy on the training set: {:.3f}'.format(log_reg001.score(x_train, y_train)))
print('Accuracy on the training set: {:.3f}'.format(log_reg001.score(x_test, y_test)))