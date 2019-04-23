from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

iris = load_iris()

#print(iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(x_train, y_train)

#print('The decision function for the 3-class iris dataset:\n\n{}'.format(gbrt.decision_function(x_test[:10])))

print('Predicted probabilities for the sample in the iris dataset:\n\n{}'.format(gbrt.predict_proba(x_test[:10])))
