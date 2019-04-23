from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import graphviz

cancer = load_breast_cancer()
#print(cancer.DESCR)

#print(cancer.feature_names)
#print(cancer.target_names)

print(cancer.data.shape)