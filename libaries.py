######
# NumPy
# Description: NumPy is a library for the Python programming language,
# adding support for large, multi-dimensional arrays and matrices,
# along with a large collection of high-level mathematical functions to operate on these array
##########
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)

######
# SciPy
# Description: SciPy is a free and open-source Python library used for scientific computing and technical computing.
# SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal
# and image processing, ODE solvers and other tasks common in science and engineering.
##########
from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print(eye)
# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)

######
# MatPlotLib
# Description: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
# It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+.
##########
import matplotlib.pyplot as plt
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
plt.show()

######
# Pandas
# Description: pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
##########
import pandas as pd
# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
print(data_pandas)

######
# SciKit Learn
# Description: scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.
##########
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

######
# MGLearn
# Description: Helper functions for the book "Introduction to Machine Learning with Python"
# Dependencies: pip3 install numpy scipy scikit-learn matplotlib pandas pillow graphviz
##########
import mglearn
import matplotlib.pyplot as plt
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plot.show()
print("X.shape: {}".format(X.shape))
