import numpy as np
import math
import scipy.stats as ss
from bokeh.plotting import figure, output_file, show
from sklearn import linear_model
from numpy.linalg import inv

# load data
data = np.loadtxt("bike_sharing.csv", delimiter=",", skiprows=1)

# normalize the data
data = ss.zscore(data)

# randomize rows and split features from labels
ndata = np.random.permutation(data)
features = ndata.shape[1]
nt = int(math.floor(len(ndata) * 0.7))
trainingFeatures = ndata[0:nt, 0:features - 1]
trainingLabels = ndata[0:nt, -1]
testFeatures = ndata[nt:, 0:features - 1]
testLabels = ndata[nt:, -1]

# add array of ones to the matrix of training features
rows = len(trainingFeatures)
arrayOfOnes = np.ones(rows)
trainingFeatures = np.c_[arrayOfOnes, trainingFeatures]

# add array of ones to the matrix of test features
rows = len(testFeatures)
arrayOfOnes = np.ones(rows)
testFeatures = np.c_[arrayOfOnes, testFeatures]

# create the initial theta array with zeros
t = np.zeros(features)


def cost_function(training_features, training_labels, theta):
    return np.sum(np.power((training_features.dot(theta) - training_labels), 2)) / (2 * len(training_features))


def gradient_descent(X, y, theta, alpha, iterations):
    temp = np.zeros(theta.shape)
    parameters = len(theta)
    size = len(X)
    cost = np.zeros(iterations)

    for i in range(iterations):
        error = y - X.dot(theta)
        for j in range(parameters):
            temp[j] = theta[j] + (alpha / size) * error.dot(X[:, j])
        theta = temp
        cost[i] = cost_function(trainingFeatures, trainingLabels, theta)
    return theta, cost


theta, cost = gradient_descent(trainingFeatures, trainingLabels, t, 0.01, 1000)


def predict(X, w):
    return X.dot(w)


print theta
# print cost_function(trainingFeatures, trainingLabels, theta)
# a = predict(testFeatures, theta)
# b = testLabels
# print zip(a,b)

# forma normal
p = inv(trainingFeatures.T.dot(trainingFeatures)).dot(trainingFeatures.T.dot(trainingLabels))

regr = linear_model.LinearRegression()
regr.fit(trainingFeatures, trainingLabels)
print("Mean squared error original: %f"
      % np.mean((regr.predict(testFeatures) - testLabels) ** 2))

print("Mean squared error minha solucao: %f"
      % np.mean((predict(testFeatures, theta) - testLabels) ** 2))

print("Mean squared error forma normal: %f"
      % np.mean((predict(testFeatures, p) - testLabels) ** 2))


it = np.arange(1000)
p = figure(x_axis_label='Iterations', y_axis_label='Cost')
p.line(it, cost, line_width=2)
show(p)
