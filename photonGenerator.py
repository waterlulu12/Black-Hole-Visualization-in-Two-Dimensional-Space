import numpy as np


def photonHeightGen(start, end, step):
    arrayLen = int((end - start) / step + 1)
    list = np.zeros(arrayLen)
    xC = start
    for x in range(0, arrayLen):
        list[x] = xC
        xC += step
    return list


def photonGen(startPoint, height, smoothness):
    xC = startPoint
    arrayLen = int((3 - startPoint) / (1 / smoothness))
    listofList = []
    start = 0
    for i, h in enumerate(height):
        list = np.zeros((arrayLen, 2))
        xC = startPoint
        for x in range(0, int(arrayLen)):
            list[x] = [xC, h]
            xC += 1 / smoothness
        listofList.append(list)
    return listofList
