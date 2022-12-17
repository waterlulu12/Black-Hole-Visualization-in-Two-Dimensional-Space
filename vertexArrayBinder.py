import OpenGL.GL as gl
import ctypes


def vertexBinder(VAO, VBO, previousSize, ploc, cloc, list):
    gl.glBindVertexArray(VAO)
    stride = list.strides[0]
    offset = ctypes.c_void_p(previousSize)
    gl.glEnableVertexAttribArray(ploc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
    gl.glVertexAttribPointer(ploc, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)

    offset = ctypes.c_void_p(previousSize + list.dtype["position"].itemsize)
    gl.glEnableVertexAttribArray(cloc)
    gl.glVertexAttribPointer(cloc, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, offset)
