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

    # Euclidian distance between two arrays
    def euclidianDistance(self, array1, array2):
        distance = [(x - y) ** 2 for x, y in zip(array1, array2)]
        distance = math.sqrt(sum(distance))
        return distance

    # Get the predictions of the k nearest neighbours
    def predict(self, test_features):
        self.predictions = []
        for i, instance in enumerate(test_features):
            neighbours = self.getNeighbours(instance, self.X, self.y, self.k)
            prediction = self.vote(neighbours)
            self.predictions.append(prediction)
        return self.predictions

    # Get the k nearest neighbours of an instance
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
        p = self.predict(test_features)
        equalValues = np.sum(p == test_labels)
        return float(equalValues) / float(len(test_labels))


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


def main():
    training_features, training_labels, test_features, test_labels = prepareData()

    k = input("Enter the number of neighbours:")
    print "\n"

    knn = KNeighbors(number_of_neighbors=k)
    knn.fit(training_features, training_labels)
    knn.predict(test_features)
    print "Leo's KNN implementation score:"
    print knn.score(test_features, test_labels)
    print "\n"

    wknn3 = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    wknn3.fit(training_features, training_labels)
    wknn3.predict(test_features)
    print "SKLEARN's KNN implementation score:"
    print wknn3.score(test_features, test_labels)


if __name__ == "__main__":
    main()
