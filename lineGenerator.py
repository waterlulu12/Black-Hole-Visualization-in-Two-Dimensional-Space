import numpy as np


def lineGen(startPoint, height, smoothness):
    xC = startPoint
    arrayLen = int((1 - startPoint) / (1 / smoothness))
    list = np.zeros((arrayLen, 2))
    for x in range(0, arrayLen):
        list[x] = [xC, height]
        xC += 1 / smoothness

    return list
