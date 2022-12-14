import ctypes
import sys
import numpy as np
import glm
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
varying vec4 v_color;
void main()
{
  gl_FragColor = v_color;
}
"""

# Build data
data = np.zeros(4, [("position", np.float32, 2), ("colors", np.float32, 4)])
data["position"] = (-0.1, +0.1), (+0.1, +0.1), (-0.1, -0.1), (+0.1, -0.1)
data["colors"] = (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1)

scaleX = 1.0
scaleY = 1.0
rotateDeg = 1.0
program = None

# Animation Control
speed = 1000
smoothness = 10000
scaleMax = 1 / 0.1
switch = 1
fps = 100
r_smoothness = 1000


def scaleAnimation(value):
    global scaleX, scaleY, rotateDeg, switch

    if scaleX + (value / smoothness) <= scaleMax and switch:
        scaleX = scaleX + (value / smoothness)
        print("hi %f", scaleX)
        scaleY = scaleY + (value / smoothness)
    elif scaleX - (value / smoothness) > 0 and not switch:
        scaleX = scaleX - (value / smoothness)
        print("hi %f", scaleX)
        scaleY = scaleY - (value / smoothness)
    elif scaleX + 0.1 > scaleMax or scaleX - 0.1 <= 0:
        switch = not switch

    rotateDeg = rotateDeg + (value / r_smoothness)
    print("hi %f", rotateDeg)

    # Scaling
    transform = glm.mat4(1)
    transform = glm.rotate(transform, glm.radians(-rotateDeg), glm.vec3(0, 0, 1.0))
    transform = glm.scale(transform, glm.vec3(scaleX, scaleY, 0.0))

    loc = gl.glGetUniformLocation(program, "transform")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))
    glut.glutTimerFunc(int(1000 / fps), scaleAnimation, speed)


def compileShader(source, type):
    shader = gl.glCreateShader(type)
    gl.glShaderSource(shader, source)

    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        print(error)
        raise RuntimeError("{source} shader compilation error")
    return shader


def createProgram(vertex, fragment):
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)

    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        print(gl.glGetProgramInfoLog(program))
        raise RuntimeError("Error Linking program")

    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    return program


def init():
    global program
    global data

    program = createProgram(
        compileShader(vertexShader, gl.GL_VERTEX_SHADER),
        compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER),
    )

    # Use Program
    gl.glUseProgram(program)
    buffer = gl.glGenBuffers(1)

    # Make this buffer the default one
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

    # Bind Data
    stride = data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    offset = ctypes.c_void_p(data.dtype["position"].itemsize)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

    # Scaling
    # transform = glm.mat4(1)
    # transform = glm.scale(transform, glm.vec3(scaleX, scaleY, 0.0))
    # loc = gl.glGetUniformLocation(program, "transform")
    # gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))

    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


def render():
    # Bind Scaling
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glLoadIdentity()

    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    glut.glutSwapBuffers()
    glut.glutPostRedisplay()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b"\x1b":
        sys.exit()


glut.glutInit()

glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)  # type: ignore #ignore
glut.glutCreateWindow("Hello world!")
glut.glutReshapeWindow(512, 512)
glut.glutReshapeFunc(reshape)
init()
glut.glutDisplayFunc(render)
scaleAnimation(speed)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

glut.glutMainLoop()
