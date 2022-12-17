from midPointCircle import midPointCircle
import numpy as np
import OpenGL.GL as gl


def blackHoleColors(n):
    return (0.0, 0.0, 0.0, 1.0)


def blackHoleRing1Colors(n):
    return (1.0, 1.0, 1.0, 1.0)


def blackHoleRing2Colors(n):
    return (1.0, 0.51, 0.0, 1.0)


def blackHoleRing4Colors(n):
    return (0.70, 0.70, 0.70, 1.0)


def blackHoleGen(xC, yC, radius, resolution):

    # Black Hole
    circleVertices = midPointCircle(xC, yC, radius, resolution)
    blackHole = np.zeros(
        len(circleVertices),
        [("position", np.float32, 2), ("color", np.float32, 4)],
    )
    blackHole["position"] = circleVertices
    blackHole["color"] = list(map(blackHoleColors, circleVertices))

    # Black Hole Ring 1
    circleVertices = midPointCircle(xC, yC, int(radius * 1.8), resolution)
    blackHoleRing1 = np.zeros(
        len(circleVertices),
        [("position", np.float32, 2), ("color", np.float32, 4)],
    )
    blackHoleRing1["position"] = circleVertices
    blackHoleRing1["color"] = list(map(blackHoleRing1Colors, circleVertices))

    # Photon Ring
    circleVertices = midPointCircle(xC, yC, int(radius * 2.4), resolution)
    photonRing = np.zeros(
        len(circleVertices),
        [("position", np.float32, 2), ("color", np.float32, 4)],
    )
    photonRing["position"] = circleVertices
    photonRing["color"] = list(map(blackHoleRing2Colors, circleVertices))

    # Black Hole Ring 2
    circleVertices = midPointCircle(xC, yC, int(radius * 3), resolution)
    blackHoleRing2 = np.zeros(
        len(circleVertices),
        [("position", np.float32, 2), ("color", np.float32, 4)],
    )
    blackHoleRing2["position"] = circleVertices
    blackHoleRing2["color"] = list(map(blackHoleColors, circleVertices))

    # Accretion Disk
    circleVertices = midPointCircle(xC, yC, int(radius * 3.5), resolution)
    accretionDisk = np.zeros(
        len(circleVertices),
        [("position", np.float32, 2), ("color", np.float32, 4)],
    )
    accretionDisk["position"] = circleVertices
    accretionDisk["color"] = list(map(blackHoleRing4Colors, circleVertices))

    return blackHole, blackHoleRing1, photonRing, blackHoleRing2, accretionDisk
