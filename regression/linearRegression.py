import numpy as np
import math
import scipy.stats as ss
from bokeh.plotting import figure, output_file, show
from sklearn import linear_model

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

# non normalized data
trFeatures = trainingFeatures
trLabels = trainingLabels
ttFeatures = testFeatures
ttLabels = testLabels

# normalize the data
trainingFeatures = ss.zscore(trainingFeatures)
testFeatures = ss.zscore(testFeatures)

# add array of ones to the matrix of training features
rows = len(trainingFeatures)
arrayOfOnes = np.ones(rows)
trainingFeatures = np.c_[arrayOfOnes, trainingFeatures]

# add array of ones to the matrix of test features
rows = len(testFeatures)
arrayOfOnes = np.ones(rows)
testFeatures = np.c_[arrayOfOnes, testFeatures]


def cost_function(training_features, training_labels, theta):
    return np.sum(np.power((training_features.dot(theta) - training_labels), 2)) / (2 * len(training_features))


def gradient_descent(X, y, alpha, iterations):
    size, features = X.shape
    theta = np.zeros(features)
    temp = theta[:]
    cost = np.zeros(iterations)

    for i in range(iterations):
        error = y - X.dot(theta)
        # for j in range(features):
        temp = theta + (alpha / size) * error.dot(X)
        theta = temp
        cost[i] = cost_function(trainingFeatures, trainingLabels, theta)
    return theta, cost


def predict(X, w):
    return X.dot(w)


# Minha solucao
weights, cost = gradient_descent(trainingFeatures, trainingLabels, 0.01, 1000)
print weights
print("Mean squared error minha solucao: %f"
      % np.mean((predict(testFeatures, weights) - testLabels) ** 2))

# SKlearn
regr = linear_model.LinearRegression(normalize=True, )
regr.fit(trFeatures, trLabels)
print regr.coef_
print("Mean squared error original: %f"
      % np.mean((regr.predict(ttFeatures) - ttLabels) ** 2))

# forma normal
# p = inv(trainingFeatures.T.dot(trainingFeatures)).dot(trainingFeatures.T.dot(trainingLabels))
# print("Mean squared error forma normal: %f"
#       % np.mean((predict(testFeatures, p) - testLabels) ** 2))

it = np.arange(1000)
p = figure(x_axis_label='Iterations', y_axis_label='Cost')
p.line(it, cost, line_width=2)
show(p)
