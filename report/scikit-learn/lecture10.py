from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mglearn
import graphviz
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
#print('Accuracy on the training set: {:.3f}'.format(log_reg001.score(x_train, y_train)))
#print('Accuracy on the training set: {:.3f}'.format(log_reg001.score(x_test, y_test)))

#mglearn.plots.plot_linear_regression_wave()
plt.plot(log_reg.coef_.T, 'o', label='C=1')
plt.plot(log_reg100.coef_.T, '^', label='C=100')
plt.plot(log_reg001.coef_.T, 'v', label='C=0.01')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.legend()
plt.show()