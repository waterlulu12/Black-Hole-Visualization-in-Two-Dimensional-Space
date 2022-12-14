import numpy as np
from toNVC import toNVC


def midPointCircle(x_center, y_center, r, res):

    x = r
    y = 0

    x_coordinates = np.array([])
    y_coordinates = np.array([])

    x_coordinates = np.append(x_coordinates, x + x_center)
    y_coordinates = np.append(y_coordinates, y + y_center)

    if r > 0:
        x_coordinates = np.append(x_coordinates, x + x_center)
        x_coordinates = np.append(x_coordinates, y + x_center)
        x_coordinates = np.append(x_coordinates, -y + x_center)
        y_coordinates = np.append(y_coordinates, -y + y_center)
        y_coordinates = np.append(y_coordinates, x + y_center)
        y_coordinates = np.append(y_coordinates, x + y_center)

    pk = 1 - r

    while x > y:

        y += 1

        if pk <= 0:
            pk = pk + 2 * y + 1

        else:
            x -= 1
            pk = pk + 2 * y - 2 * x + 1

        if x < y:
            break

        x_coordinates = np.append(x_coordinates, x + x_center)
        x_coordinates = np.append(x_coordinates, -x + x_center)
        x_coordinates = np.append(x_coordinates, x + x_center)
        x_coordinates = np.append(x_coordinates, -x + x_center)
        x_coordinates = np.append(
            x_coordinates,
            y + x_center,
        )
        x_coordinates = np.append(x_coordinates, -y + x_center)
        x_coordinates = np.append(x_coordinates, y + x_center)
        x_coordinates = np.append(x_coordinates, -y + x_center)
        y_coordinates = np.append(y_coordinates, y + y_center)
        y_coordinates = np.append(y_coordinates, y + y_center)
        y_coordinates = np.append(y_coordinates, -y + y_center)
        y_coordinates = np.append(y_coordinates, -y + y_center)
        y_coordinates = np.append(y_coordinates, x + y_center)
        y_coordinates = np.append(y_coordinates, x + y_center)
        y_coordinates = np.append(y_coordinates, -x + y_center)
        y_coordinates = np.append(y_coordinates, -x + y_center)

    return toNVC(x_coordinates, y_coordinates, res)
