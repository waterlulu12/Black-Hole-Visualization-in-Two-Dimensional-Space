import ctypes
import sys
import numpy as np
import glm
from compileShader import compileShader
from createProgram import createProgram
from midPointCircle import midPointCircle
from lineGenerator import lineGen, lineHeightGen
import OpenGL.GL as gl
import OpenGL.GLUT as glut

# Define Shaders
vertexShader = """
attribute vec4 color;
attribute vec2 position;
varying vec4 v_color;
uniform mat4 transform;
void main()
{
  gl_Position = transform*vec4(position, 0.0, 1.0);
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

# Animation Control
speed = 10
smoothness = 1000
scaleMax = 1 / 0.1
switch = 1
fps = 100
r_smoothness = 1000
tx = 0
ty = 0


def scaleAnimation(value):
    global scaleX, scaleY, rotateDeg, switch, tx, ty

    tx -= 1 / smoothness

    transform = glm.mat4(1)
    transform = glm.translate(transform, glm.vec3(tx, ty, 0))

    loc = gl.glGetUniformLocation(program, "transform")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))

    glut.glutTimerFunc(int(1000 / fps), scaleAnimation, speed)


def blackHoleColors(n):
    return (0.0, 0.0, 0.0, 0.0)


def lineColors(n):
    return (1.0, 0.0, 0.0, 0.0)


circleVertices = midPointCircle(0, 0, 1000, 6000)
blackHole = np.zeros(
    len(circleVertices),
    [("position", np.float32, 2), ("color", np.float32, 4)],
)
blackHole["position"] = circleVertices
blackHole["color"] = list(map(blackHoleColors, circleVertices))

lineHeights = lineHeightGen(0.5, 0.9, 0.05)

lineVertices = lineGen(0.2, lineHeights, 100)

lightRay = np.zeros(
    len(lineVertices),
    [("position", np.float32, 2), ("color", np.float32, 4)],
)
lightRay["position"] = lineVertices
lightRay["color"] = list(map(lineColors, lineVertices))


lightIndices = np.array([], dtype=np.int32)
np.set_printoptions(suppress=True)
for x in range(1, len(lightRay)):
    if (x % (int(len(lightRay) / len(lineHeights)))) != 0:
        lightIndices = np.append(lightIndices, int(x - 1))
        lightIndices = np.append(lightIndices, int(x))


blackHoleVAO = None
lightRayVAO = None


def init():
    global program, blackHoleVAO, lightRayVAO

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
        lightRay.nbytes + blackHole.nbytes,
        lightRay,
        gl.GL_STATIC_DRAW,
    )
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, lightRay.nbytes, lightRay)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, lightRay.nbytes, blackHole.nbytes, blackHole)

    # Generate Indices Buffer
    indicesBuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indicesBuffer)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, lightIndices.nbytes, lightIndices, gl.GL_STATIC_DRAW
    )

    lightRayVAO = gl.glGenVertexArrays(1)
    blackHoleVAO = gl.glGenVertexArrays(1)

    # Get Shader Attributes
    ploc = gl.glGetAttribLocation(program, "position")
    cloc = gl.glGetAttribLocation(program, "color")

    # Bind Light

    gl.glBindVertexArray(lightRayVAO)
    stride = lightRay.strides[0]
    offset = ctypes.c_void_p(0)
    gl.glEnableVertexAttribArray(ploc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glVertexAttribPointer(ploc, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)

    offset = ctypes.c_void_p(lightRay.dtype["position"].itemsize)
    gl.glEnableVertexAttribArray(cloc)
    gl.glVertexAttribPointer(cloc, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indicesBuffer)

    # Bind Black Hole
    stride = blackHole.strides[0]
    offset = ctypes.c_void_p(lightRay.nbytes)
    gl.glBindVertexArray(blackHoleVAO)
    gl.glEnableVertexAttribArray(ploc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glVertexAttribPointer(ploc, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)

    offset = ctypes.c_void_p(lightRay.nbytes + blackHole.dtype["position"].itemsize)
    gl.glEnableVertexAttribArray(cloc)
    gl.glVertexAttribPointer(cloc, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)


def render():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glLineWidth(2)

    gl.glBindVertexArray(lightRayVAO)
    gl.glDrawElements(gl.GL_LINES, len(lightIndices), gl.GL_UNSIGNED_INT, None)

    gl.glLoadIdentity()
    gl.glBindVertexArray(blackHoleVAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(blackHole))

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
glut.glutReshapeWindow(800, 800)
glut.glutReshapeFunc(reshape)
init()
scaleAnimation(speed)
glut.glutDisplayFunc(render)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

glut.glutMainLoop()
