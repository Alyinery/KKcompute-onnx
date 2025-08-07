import subprocess
import tempfile
import os


def compile_source(glsl_code):
    """
    Compile GLSL to SpirV and return as bytes.
    """

    if not isinstance(glsl_code, str):
        raise TypeError("glslangValidator expects a string.")

    filename1 = os.path.join(tempfile.gettempdir(), "x.txt")
    filename2 = os.path.join(tempfile.gettempdir(), "x.spv")

    with open(filename1, "wb") as f:
        f.write(glsl_code.encode())

    # Note: -O means optimize, use -O0 to disable optimization
    try:
        stdout = subprocess.check_output(
            ["glslangValidator", "-S", "comp", "-V", "-Os", "-o", filename2, filename1], stderr=subprocess.STDOUT
        )
        stdout  # noqa - not used
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    with open(filename2, "rb") as f:
        binary = f.read()

    return binary
