import numpy as np
import math
import heapq
from sklearn.neighbors import KNeighborsClassifier


class KNeighbors():

    def __init__(self, number_of_neighbors=3):
        self.k = number_of_neighbors
        self.predictions = []

    def fit(self, training_features, training_labels):
        self.X = training_features
        self.y = training_labels

    def euclidianDistance(self, array1, array2):
        distance = [(x - y) ** 2 for x, y in zip(array1, array2)]
        distance = math.sqrt(sum(distance))
        return distance

    def predict(self, test_features):
        for i, instance in enumerate(test_features):
            neighbours = self.getNeighbours(instance, self.X, self.y, self.k)
            prediction = self.vote(neighbours)
            self.predictions.append(prediction)
        return self.predictions

    def getNeighbours(self, instance, training_features, training_labels, k):
        distanceList = []
        size, columns = training_features.shape
        for i in xrange(size):
            # calculating the distance between the instance and each element of the trainingSet.
            # The distance measure does not include the label.
            distanceList.append(
                (
                    self.euclidianDistance(instance, training_features[i]),
                    np.append(training_features[i],training_labels[i])))

        # getting the k smallest neighbours of the instance
        result = heapq.nsmallest(k, distanceList, key=lambda x: x[0])
        _, neighbours = zip(*result)
        return neighbours

    def vote(self, neighbours):
        votes = []
        for neighbour in neighbours:
            votes.append(neighbour[-1])
        return max(set(votes), key=votes.count)

    def score(self, test_features, test_labels):
        p = self.predictions
        equalValues = np.sum(p == test_labels)
        return float(equalValues) / float(len(test_labels))


# def prepareData():
#     data = np.loadtxt("haberman.data", delimiter=",")
#     ndata = np.random.permutation(data)
#     size = len(ndata)
#     nt = int(math.floor(size * 0.7))
#     trainingSet = ndata[0:nt]
#     testSet = ndata[nt:size]
#     return trainingSet, testSet
#
#
# def euclidianDistance(array1, array2):
#     distance = [(x - y) ** 2 for x, y in zip(array1, array2)]
#     distance = math.sqrt(sum(distance))
#     return distance
#
#
# def getNeighbours(instance, trainingSet, k):
#     distanceList = []
#     size, columns = trainingSet.shape
#     for i in xrange(size):
#         # calculating the distance between the instance and each element of the trainingSet.
#         # The distance measure does not include the label.
#         distanceList.append((euclidianDistance(instance[0:columns - 1], trainingSet[i, 0:columns - 1]), trainingSet[i]))
#
#     # getting the k smallest neighbours of the instance
#     result = heapq.nsmallest(k, distanceList, key=lambda x: x[0])
#     _, neighbours = zip(*result)
#     return neighbours
#
#
# def vote(neighbours):
#     predictions = []
#     for neighbour in neighbours:
#         predictions.append(neighbour[-1])
#     return max(set(predictions), key=predictions.count)
#
#
# def predict(trainingSet, testSet, k):
#     predictions = []
#     for i, instance in enumerate(testSet):
#         neighbours = getNeighbours(instance, trainingSet, k)
#         prediction = vote(neighbours)
#         predictions.append(prediction)
#     return predictions
#
#
# def score(prediction, testSet):
#     equalValues = np.sum(prediction == testSet)
#     print equalValues
#     print len(testSet)
#     return float(equalValues) / float(len(testSet))

def prepareData():
    data = np.loadtxt("haberman.data", delimiter=",")
    ndata = np.random.permutation(data)
    size, features = ndata.shape
    nt = int(math.floor(size * 0.7))

    training_features = ndata[0:nt, 0:features-1]
    training_labels = ndata[0:nt, -1]
    test_features = ndata[nt:size, 0:features-1]
    test_labels = ndata[nt:size, -1]
    return training_features, training_labels, test_features, test_labels


training_features, training_labels, test_features, test_labels = prepareData()
knn = KNeighbors(number_of_neighbors=5)
knn.fit(training_features, training_labels)
knn.predict(test_features)
print knn.score(test_features, test_labels)

wknn3 = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
wknn3.fit(training_features, training_labels)
wknn3.predict(test_features)
print wknn3.score(test_features, test_labels)
