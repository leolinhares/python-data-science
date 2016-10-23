import numpy as np
import math
import heapq
from sklearn.neighbors import KNeighborsClassifier


def prepareData():
    data = np.loadtxt("haberman.data", delimiter=",")
    ndata = np.random.permutation(data)
    size = len(ndata)
    nt = int(math.floor(size * 0.7))
    trainingSet = ndata[0:nt]
    testSet = ndata[nt:size]
    # trlabels = ndata[0:nt, 3]
    # ttlabels = ndata[nt:size, 3]
    return trainingSet, testSet


def euclidianDistance(array1, array2):
    distance = [(x - y) ** 2 for x, y in zip(array1, array2)]
    distance = math.sqrt(sum(distance))
    return distance


def getNeighbours(instance, trainingSet, k):
    distanceList = []
    for i in xrange(len(trainingSet)):
        # calculating the distance between the instance and each element of the trainingSet.
        # The distance measure does not include the label.
        distanceList.append((euclidianDistance(instance[0:3], trainingSet[i,0:3]), trainingSet[i]))

    # getting the k smallest neighbours of the instance
    result = heapq.nsmallest(k, distanceList, key=lambda x: x[0])
    _, neighbours = zip(*result)
    return neighbours

def predict(neighbours):
    predictions = []
    for neighbour in neighbours:
        predictions.append(neighbour[-1])
    return max(set(predictions), key=predictions.count)

def fit(trainingSet, testSet, k):
    predictions = []
    for i, instance in enumerate(testSet):
        neighbours = getNeighbours(instance, trainingSet, k)
        prediction = predict(neighbours)
        predictions.append(prediction)
    return predictions

def score(prediction, testSet):
    equalValues = np.sum(prediction == testSet)
    print equalValues
    print len(testSet)
    return float(equalValues)/float(len(testSet))

trainingSet, testSet = prepareData()
predictions = fit(trainingSet,testSet,3)
print score(predictions, testSet[:,-1])

wknn3 = KNeighborsClassifier(n_neighbors=3,weights='uniform')
wknn3.fit(trainingSet[:,0:3], trainingSet[:,-1])
print wknn3.score(testSet[:,0:3],testSet[:,-1])