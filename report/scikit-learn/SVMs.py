from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC()
svm.fit(x_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(svm.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(svm.score(x_test, y_test)))

"""plt.plot(x_train.min(axis=0), 'o', label='Min')
plt.plot(x_train.max(axis=0), 'v', label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature Magnitude in Log Scale')
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()"""

min_train = x_train.min(axis=0)
range_train = (x_train - min_train).max(axis=0)

x_train_scaled = (x_train - min_train) / range_train

#print('Minimum per feature\n{}'.format(x_train_scaled.min(axis=0)))
#print('Maximum per feature\n{}'.format(x_train_scaled.max(axis=0)))

x_test_scaled = (x_test - min_train) / range_train

svm = SVC(C=1000)
svm.fit(x_train_scaled, y_train)

#print('Accuracy on the training subset: {:.3f}'.format(svm.score(x_train_scaled, y_train)))
#print('Accuracy on the test subset: {:.3f}'.format(svm.score(x_test_scaled, y_test)))

#print('The decision function is:\n\n{}'.format(svm.decision_function(x_test_scaled)[:20]))
#print('Thresholded decision function:\n\n{}'.format(svm.decision_function(x_test_scaled)[:20] > 0))

svm = SVC(C=1000, probability=True)
svm.fit(x_train_scaled, y_train)

print('Predicted probabilities for the sample (malignant and benign):\n\n{}'.format(svm.predict_proba(x_test_scaled[:20])))
# 1이면 양성 0이면 악성
print(svm.predict(x_test_scaled)[:20])