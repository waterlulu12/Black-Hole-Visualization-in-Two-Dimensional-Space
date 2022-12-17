import sys
import numpy as np
from compileShader import compileShader
from createProgram import createProgram
from blackHole import blackHoleGen
from photonGenerator import photonGen, photonHeightGen
from vertexArrayBinder import vertexBinder
import OpenGL.GL as gl
import OpenGL.GLUT as glut

# Define Shaders
vertexShader = """
attribute vec4 color;
attribute vec2 position;
varying vec4 v_color;
void main()
{
  gl_Position = vec4(position, 0.0, 1.0);
  v_color= color;
}
"""

fragmentShader = """
varying  vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""


def lineColors(n):
    return (1.0, 0.0, 0.0, 1.0)


# Arbitary Black Hole Constants
c = 3
G = 6
M = 13
rs = (2 * 10 * G * M) / (c * c)

resolution = 800

(
    blackHole,
    blackHoleRing1,
    photonRing,
    blackHoleRing3,
    accretionDisk,
) = blackHoleGen(0, 0, rs, resolution)

lightHeights = photonHeightGen(0.0, 0.98, 0.1)
lightRayVertices = photonGen(0.8, lightHeights, 100)


lightRay = np.zeros(
    len(lightRayVertices[0]) * len(lightHeights),
    [("position", np.float32, 2), ("color", np.float32, 4)],
)

tempVertices = []
for i, x in enumerate(lightRayVertices):
    for j, y in enumerate(lightRayVertices[i]):
        tempVertices.append(lightRayVertices[i][j])

# print(tempVertices)
lightRay["position"] = tempVertices
lightRay["color"] = list(map(lineColors, lightRay["position"]))

# lightIndices = np.array([], dtype=np.int32)

# for x in range(0, len(lightRay)):
#     lightIndices = np.append(lightIndices, x)

blackHoleVAO = None
blackHoleRing1VAO = None
blackHoleRing2VAO = None
blackHoleRing3VAO = None
blackHoleRing4VAO = None
lightRayVAO = None
vertexBuffer = None
indicesBuffer = None


# Animation Control
fps = 30
speed = 100
smoothness = 0.0001


def blackHoleAnimation(value):
    global scaleX, scaleY, rotateDeg, switch, tx, ty, lightRayVertices

    arbitaryRadius = rs / resolution
    arbitaryMass = (arbitaryRadius * (c * c)) / (2 * (G))
    arbitarySpeed = np.sqrt((arbitaryRadius / (2 * G * M)))
    arbitarySpeed = arbitarySpeed * 1.4
    arbitaryG = (arbitaryRadius * arbitarySpeed * arbitarySpeed) / (2 * arbitaryMass)
    arbitaryG = arbitaryG * 1200
    # send new data
    for i, x in enumerate(tempVertices):
        dist = np.linalg.norm([0.0, 0.0] - tempVertices[i])
        d2 = np.array([0.0, arbitaryRadius * 0.92])
        d3 = np.array([0.0, arbitaryRadius * 0.9])
        normal_distance = np.linalg.norm([0.0, 0.0] - d2)
        normal_distance = abs(normal_distance)
        second_distance = np.linalg.norm([0.0, 0.0] - d3)

        if dist != 0.0:
            if dist > normal_distance:
                factor = arbitaryG * arbitaryMass / (dist * dist)
                factor = factor * 0.1
                tempVertices[i][0] -= arbitarySpeed + (tempVertices[i][0] * factor)
                tempVertices[i][1] -= tempVertices[i][1] * factor
            else:
                tempVertices[i][0] = 0
                tempVertices[i][1] = 0

    # print(tempVertices)
    lightRay["position"] = tempVertices

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes
        + blackHole.nbytes,
        lightRay.nbytes,
        lightRay,
    )

    glut.glutTimerFunc(int(1000 / fps), blackHoleAnimation, speed)


def init():
    global program, blackHoleVAO, lightRayVAO, blackHoleRing1VAO, blackHoleRing2VAO, blackHoleRing3VAO, blackHoleRing4VAO, vertexBuffer, indicesBuffer

    program = createProgram(
        compileShader(vertexShader, gl.GL_VERTEX_SHADER),
        compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER),
    )

    gl.glUseProgram(program)

    # Generate Vertex Buffer

    vertexBuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes
        + blackHole.nbytes
        + lightRay.nbytes,
        accretionDisk,
        gl.GL_DYNAMIC_DRAW,
    )
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, accretionDisk.nbytes, accretionDisk)
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER, accretionDisk.nbytes, blackHoleRing3.nbytes, blackHoleRing3
    )
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes + blackHoleRing3.nbytes,
        photonRing.nbytes,
        photonRing,
    )
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes + blackHoleRing3.nbytes + photonRing.nbytes,
        blackHoleRing1.nbytes,
        blackHoleRing1,
    )
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes,
        blackHole.nbytes,
        blackHole,
    )
    gl.glBufferSubData(
        gl.GL_ARRAY_BUFFER,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes
        + blackHole.nbytes,
        lightRay.nbytes,
        lightRay,
    )

    blackHoleRing4VAO = gl.glGenVertexArrays(1)
    blackHoleRing3VAO = gl.glGenVertexArrays(1)
    blackHoleRing2VAO = gl.glGenVertexArrays(1)
    blackHoleRing1VAO = gl.glGenVertexArrays(1)
    blackHoleVAO = gl.glGenVertexArrays(1)
    lightRayVAO = gl.glGenVertexArrays(1)

    # Get Attrib Locatons
    ploc = gl.glGetAttribLocation(program, "position")
    cloc = gl.glGetAttribLocation(program, "color")

    vertexBinder(blackHoleRing4VAO, vertexBuffer, 0, ploc, cloc, accretionDisk)

    vertexBinder(
        blackHoleRing3VAO,
        vertexBuffer,
        accretionDisk.nbytes,
        ploc,
        cloc,
        blackHoleRing3,
    )

    vertexBinder(
        blackHoleRing2VAO,
        vertexBuffer,
        accretionDisk.nbytes + blackHoleRing3.nbytes,
        ploc,
        cloc,
        photonRing,
    )

    vertexBinder(
        blackHoleRing1VAO,
        vertexBuffer,
        accretionDisk.nbytes + blackHoleRing3.nbytes + photonRing.nbytes,
        ploc,
        cloc,
        blackHoleRing1,
    )

    vertexBinder(
        blackHoleVAO,
        vertexBuffer,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes,
        ploc,
        cloc,
        blackHole,
    )

    vertexBinder(
        lightRayVAO,
        vertexBuffer,
        accretionDisk.nbytes
        + blackHoleRing3.nbytes
        + photonRing.nbytes
        + blackHoleRing1.nbytes
        + blackHole.nbytes,
        ploc,
        cloc,
        lightRay,
    )


def render():
    global lightRay
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glPointSize(3)

    # transform = glm.mat4(1)
    # loc = gl.glGetUniformLocation(program, "transform")
    # gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))

    gl.glBindVertexArray(blackHoleRing4VAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(accretionDisk))

    gl.glBindVertexArray(blackHoleRing3VAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(blackHoleRing3))

    gl.glBindVertexArray(blackHoleRing2VAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(photonRing))

    gl.glBindVertexArray(blackHoleRing1VAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(blackHoleRing1))

    gl.glBindVertexArray(blackHoleVAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(blackHole))

    gl.glBindVertexArray(lightRayVAO)
    gl.glDrawArrays(gl.GL_POINTS, 0, len(lightRay))

    glut.glutSwapBuffers()
    glut.glutPostRedisplay()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b"\x1b":
        sys.exit()


glut.glutInit()

glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)  # type: ignore #ignore
glut.glutCreateWindow("Black Hole Visualization")
glut.glutReshapeWindow(resolution, resolution)
glut.glutReshapeFunc(reshape)
init()
glut.glutDisplayFunc(render)
blackHoleAnimation(speed)
glut.glutKeyboardFunc(keyboard)

glut.glutMainLoop()
