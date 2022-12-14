import ctypes
import sys
import numpy as np
import glm
from compileShader import compileShader
from createProgram import createProgram
from midPointCircle import midPointCircle
from lineGenerator import lineGen
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

# Build data
# data = np.zeros(4, [("position", np.float32, 2), ("colors", np.float32, 4)])
# data["position"] = (-0.1, +0.1), (+0.1, +0.1), (-0.1, -0.1), (+0.1, -0.1)
# data["colors"] = (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1)


# scaleX = 1.0
# scaleY = 1.0
# rotateDeg = 1.0
# program = None

# Animation Control
# speed = 1000
# smoothness = 10000
# scaleMax = 1 / 0.1
# switch = 1
# fps = 100
# r_smoothness = 1000


# def scaleAnimation(value):
#     global scaleX, scaleY, rotateDeg, switch

#     if scaleX + (value / smoothness) <= scaleMax and switch:
#         scaleX = scaleX + (value / smoothness)
#         scaleY = scaleY + (value / smoothness)
#     elif scaleX - (value / smoothness) > 0 and not switch:
#         scaleX = scaleX - (value / smoothness)
#         scaleY = scaleY - (value / smoothness)
#     elif scaleX + 0.1 > scaleMax or scaleX - 0.1 <= 0:
#         switch = not switch

#     rotateDeg = rotateDeg + (value / r_smoothness)

#     # Scaling
#     transform = glm.mat4(1)
#     transform = glm.rotate(transform, glm.radians(-rotateDeg), glm.vec3(0, 0, 1.0))
#     transform = glm.scale(transform, glm.vec3(scaleX, scaleY, 0.0))

#     loc = gl.glGetUniformLocation(program, "transform")
#     gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))
#     glut.glutTimerFunc(int(1000 / fps), scaleAnimation, speed)


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

lineVertices = lineGen(0.5, 0.8, 100)
lightRay = np.zeros(
    len(lineVertices),
    [("position", np.float32, 2), ("color", np.float32, 4)],
)
lightRay["position"] = lineVertices
lightRay["color"] = list(map(lineColors, lineVertices))

blackHoleVBO = None
lightBuffer = None
ploc = None
cloc = None


def init():
    global program
    global blackHole
    global blackHoleVBO
    global lightBuffer
    global ploc
    global cloc

    program = createProgram(
        compileShader(vertexShader, gl.GL_VERTEX_SHADER),
        compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER),
    )

    gl.glUseProgram(program)

    blackHoleVAO = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(blackHoleVAO)

    blackHoleVBO = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, blackHoleVBO)

    stride = blackHole.strides[0]
    offset = ctypes.c_void_p(0)
    ploc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(ploc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, blackHoleVBO)
    gl.glVertexAttribPointer(ploc, 3, gl.GL_FLOAT, False, stride, offset)

    offset = ctypes.c_void_p(blackHole.dtype["position"].itemsize)
    cloc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(cloc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, blackHoleVBO)
    gl.glVertexAttribPointer(cloc, 4, gl.GL_FLOAT, False, stride, offset)

    gl.glBufferData(gl.GL_ARRAY_BUFFER, blackHole.nbytes, blackHole, gl.GL_STATIC_DRAW)


def render():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(blackHole))

    lightVAO = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(lightVAO)
    # lightBuffer = gl.glGenBuffers(1)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, lightBuffer)

    # stride = lightRay.strides[0]
    # offset = ctypes.c_void_p(0)
    # gl.glEnableVertexAttribArray(ploc)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, lightBuffer)
    # gl.glVertexAttribPointer(ploc, 3, gl.GL_FLOAT, False, stride, offset)

    # offset = ctypes.c_void_p(lightRay.dtype["position"].itemsize)
    # gl.glEnableVertexAttribArray(cloc)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, lightBuffer)
    # gl.glVertexAttribPointer(cloc, 4, gl.GL_FLOAT, False, stride, offset)
    # gl.glBufferData(gl.GL_ARRAY_BUFFER, lightRay.nbytes, lightRay, gl.GL_STATIC_DRAW)

    gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(lightRay))

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
glut.glutDisplayFunc(render)
# scaleAnimation(speed)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

glut.glutMainLoop()
