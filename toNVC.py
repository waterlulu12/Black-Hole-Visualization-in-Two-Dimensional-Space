import numpy as np


def toNVC(xList, yList, resolution):
    i = 0
    for x in xList:
        xList[i] = (xList[i]) / resolution
        yList[i] = (yList[i]) / resolution
        i += 1

    coordinateList = np.zeros((len(xList), 2))
    i = 0
    for x in xList:
        coordinateList[i] = [xList[i], yList[i]]
        i += 1
    return coordinateList
