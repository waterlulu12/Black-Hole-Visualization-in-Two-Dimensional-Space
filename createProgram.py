import OpenGL.GL as gl


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
