import numpy as np
import math
import scipy.stats as ss
from sklearn import linear_model

data = np.loadtxt("bike_sharing.csv", delimiter=",", skiprows=1)
ndata = np.random.permutation(data)
size = len(ndata)
nt = int(math.floor(size * 0.7))
trainingFeatures = ndata[0:nt, :-1]
p = trainingFeatures
testFeatures = ndata[nt:size, :-1]
trainingLabels = ndata[0:nt, -1]
testLabels = ndata[nt:size, -1]

# normalize the training set
trainingFeatures = ss.zscore(trainingFeatures, axis=1)

# add array of ones to the matrix of training features
arrayOfOnes = np.ones(len(trainingFeatures))
trainingFeatures = np.c_[arrayOfOnes, trainingFeatures]

# create the initial theta array with zeros
featuresSize = len(trainingFeatures[0])
theta = np.zeros(featuresSize)
theta = theta[:,None]

# column of training labels
trainingLabels = trainingLabels[:,None]
alpha = 0.01

for i, x in enumerate(trainingFeatures):
    theta = theta + alpha*(trainingLabels[i] - x.dot(theta))*x[:,None]

print theta
