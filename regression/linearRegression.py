import numpy as np
import math
import scipy.stats as ss
from bokeh.plotting import figure, output_file, show, save
from sklearn import linear_model
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


class LR:

    predictions = []
    weights = []
    cost = []

    def __init__(self, alpha, epochs, algorithm='stochastic', normalize=True):
        self.alpha = alpha
        self.epochs = epochs
        self.algorithm = algorithm
        self.normalize = normalize

    def add_column_of_ones(self, X):
        rows = len(X)
        arrayOfOnes = np.ones(rows)
        X = np.c_[arrayOfOnes, X]  # add column of ones
        return X

    def normalization(self, X):
        X = ss.zscore(X)
        return self.add_column_of_ones(X)

    def predict(self, X):
        if self.normalize:
            X = self.normalization(X)
        else:
            X = self.add_column_of_ones(X)
        self.predictions = X.dot(self.weights)
        return self.predictions

    def r_squared(self, X, y, weights):
        return np.sum(np.power((X.dot(weights) - y), 2)) / (2 * len(X))

    def fit(self, X, y):
        if self.algorithm == 'stochastic':
            self.stochastic(X,y)
        else:
            pass
        # elif self.algorithm == 'batch':
        #     self.batch(X, y)
        # else:
        #     pass

    def stochastic(self, X, y):
        if self.normalize:
            X = self.normalization(X)
        else:
            X = self.add_column_of_ones(X)
        size, features = X.shape
        weights = np.zeros(features)
        cost = np.zeros(self.epochs)

        for i in range(self.epochs):
            for index, row in enumerate(X):
                error = row.dot(weights) - y[index]
                weights = weights - self.alpha * error * row
            cost[i] = self.r_squared(X, y, weights)

        self.weights = weights[:]
        self.cost = cost[:]

    # def batch(self, X, y):
    #     X = self.normalization(X)
    #     size, features = X.shape
    #     weights = np.zeros(features)
    #     cost = np.zeros(self.epochs)
    #
    #     for i in range(self.epochs):
    #         error = X.dot(weights) - y
    #         weights = weights - (self.alpha/size)*error.dot(X)
    #         cost[i] = self.r_squared(X, y, weights)
    #
    #     self.weights = weights[:]
    #     self.cost = cost[:]


def get_data():

    # load data
    data = np.loadtxt("bike_sharing.csv", delimiter=",", skiprows=1)

    # randomize rows and split features from labels
    ndata = np.random.permutation(data)
    columns = ndata.shape[1]
    nt = int(math.floor(len(ndata) * 0.7))
    trainingFeatures = ndata[0:nt, 0:columns - 1]
    trainingLabels = ndata[0:nt, -1]
    testFeatures = ndata[nt:, 0:columns - 1]
    testLabels = ndata[nt:, -1]

    return trainingFeatures, trainingLabels, testFeatures, testLabels


def main():
    trainingFeatures, trainingLabels, testFeatures, testLabels = get_data()

    # SKLEARN LINEAR REGRESSION
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(trainingFeatures, trainingLabels)
    print regr.coef_
    print("Mean squared error of SKLEARN regression: %f"
          % np.mean((regr.predict(testFeatures) - testLabels) ** 2))

    s_alpha = 0.0001
    s_epochs = 20
    stochastic_regression = LR(s_alpha, s_epochs, algorithm='stochastic', normalize=True)
    stochastic_regression.fit(trainingFeatures, trainingLabels)
    print stochastic_regression.weights
    print("Mean squared error of Leo's stochastic regression (alpha=%f, epochs=%d): %f"
              % (s_alpha,  s_epochs, np.mean((stochastic_regression.predict(testFeatures) - testLabels) ** 2)))

    # Graph cost vs epochs
    it = np.arange(s_epochs)
    p = figure(x_axis_label='Epochs', y_axis_label='Cost')
    p.line(it, stochastic_regression.cost, line_width=2)
    show(p)

    # b_alpha = 0.01
    # b_epochs = 500
    # batch_regression = LR(b_alpha, b_epochs)
    # batch_regression.batch(trainingFeatures, trainingLabels)
    # print batch_regression.weights
    # print("Mean squared error batch regression (alpha=%f, epochs=%d): %f"
    #       % (b_alpha, b_epochs, np.mean((batch_regression.predict(testFeatures) - testLabels) ** 2)))
    #
    # it = np.arange(b_epochs)
    # a = figure(x_axis_label='Iterations', y_axis_label='Cost')
    # a.line(it, batch_regression.cost, line_width=2)
    # show(a)


if __name__ == "__main__":
    main()
