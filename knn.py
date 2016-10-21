import numpy as np
import math
import heapq

def prepareData():
	data = np.loadtxt("haberman.data",delimiter=",")
	ndata = np.random.permutation(data)
	size = len(ndata)
	nt = int(math.floor(size*0.7))
	trfeatures = ndata[0:nt,0:3]
	ttfeatures = ndata[nt:size,0:3]
	trlabels = ndata[0:nt,3]
	ttlabels = ndata[nt:size,3]
	return trfeatures, trlabels, ttfeatures, ttlabels

def euclidianDistance(array1, array2):
	distance = [(x - y)**2 for x, y in zip(array1, array2)]
	distance = math.sqrt(sum(distance))
	return distance

def getNeighbours(instance, trainingSet, k):
	distanceList = []
	neighbours = range(k)
	for i in xrange(len(trainingSet)):
		distanceList.append((euclidianDistance(instance, trainingSet[i]), i))
	result = heapq.nsmallest(k, distanceList)
	_, index = zip(*result)
	return index

# trfeatures,_,_,_ = prepareData()
# print getNeighbours(trfeatures[0], trfeatures[1:],3)

# def predict()

# a = [x for x in range(1,10)]
# b = [x**2 for x in range(5,20)]
# print a
# print b
# print zip(a,b)
# print euclidianDistance(a,b)
