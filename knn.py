# Required Packages
import sys
import numpy as np
import scipy as sp
import matplotlib
import pandas as pd
import sklearn

# Application specific functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data and model
iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)

# Fit Model
knn.fit(X_train, y_train)

# Output result
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
