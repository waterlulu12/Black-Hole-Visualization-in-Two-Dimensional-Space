import numpy as np


def lineHeightGen(start, end, step):
    arrayLen = int((end - start) / step + 1)
    list = np.zeros(arrayLen)
    xC = start
    for x in range(0, arrayLen):
        list[x] = xC
        xC += step
    return list


def lineGen(startPoint, height, smoothness):
    xC = startPoint
    arrayLen = int((4 - startPoint) / (1 / smoothness)) * len(height)
    list = np.zeros((arrayLen - len(height) + 1, 2))
    start = 0
    for h in height:
        xC = startPoint
        for x in range(0, int(arrayLen / len(height))):
            list[x + start] = [xC, h]
            xC += 1 / smoothness
        start += x
    return list
