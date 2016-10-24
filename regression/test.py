import numpy as np
import math
import scipy.stats as ss
from bokeh.plotting import figure, output_file, show
from sklearn import linear_model

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y, alpha, iterations):
        rows, columns = X.shape

        # add array of ones to the matrix of training features
        X = np.c_[np.ones(rows), X]

        # create the initial theta array with zeros
        theta = np.zeros(columns)

        theta = gradient_descent(X, y, theta, alpha, iterations)

        return theta

    def __gradient_descent(self, X, y, theta, alpha, iterations):
        temp = np.zeros(theta.shape)
        parameters = len(theta)
        size = len(X)

        for i in range(iterations):
            error = y - X.dot(theta)
            for j in range(parameters):
                temp[j] = theta[j] + (alpha / size) * error.dot(X[:, j])
            theta = temp
        return theta