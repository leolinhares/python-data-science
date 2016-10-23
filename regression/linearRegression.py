import numpy as np
import math

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