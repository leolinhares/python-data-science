import numpy as np
import math
import scipy.stats as ss
from bokeh.plotting import figure, output_file, show


# load data
data = np.loadtxt("bike_sharing.csv", delimiter=",", skiprows=1)

# normalize the data
data = ss.zscore(data)

# randomize rows and split features from labels
ndata = np.random.permutation(data)
features = ndata.shape[1]
trainingFeatures = ndata[:, 0:features - 1]
trainingLabels = ndata[:, -1]

# add array of ones to the matrix of training features
rows = len(trainingFeatures)
arrayOfOnes = np.ones(rows)
trainingFeatures = np.c_[arrayOfOnes, trainingFeatures]

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
            temp[j] = theta[j] + (alpha/size)*(error).dot(X[:,j])
        theta = temp
        cost[i] = cost_function(trainingFeatures,trainingLabels, theta)
    return theta,cost

x,cost = gradient_descent(trainingFeatures, trainingLabels, t, 0.01, 1000)

print x
print cost
print cost_function(trainingFeatures,trainingLabels,x)

it = np.arange(1000)
p = figure(x_axis_label='Iterations', y_axis_label='Cost')
p.line(it, cost, line_width=2)
show(p)

