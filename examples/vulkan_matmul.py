import kp
import os
import numpy as np
import subprocess
import tempfile
import time


def glsl2spirv(glsl_code):
    """
    Compile GLSL to SpirV and return as bytes.
    """

    if not isinstance(glsl_code, str):
        raise TypeError("glsl2spirv expects a string.")

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


# 初始化
device_id = 1
mgr = kp.Manager(device_id)
print(mgr.list_devices()[device_id])

# 创建输入矩阵
edge_size = 1024
a = np.random.rand(edge_size, edge_size).astype(np.float32)
b = np.random.rand(edge_size, edge_size).astype(np.float32)
t_a = mgr.tensor(a.flatten())
t_b = mgr.tensor(b.flatten())
t_out = mgr.tensor(np.zeros(edge_size*edge_size, dtype=np.float32))

# 矩阵乘法着色器
matmul_shader_code = """
#version 450
layout(local_size_x = {edge_size}, local_size_y = {edge_size}) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer C { float c[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    float sum = 0.0;
    for (uint k = 0; k < {edge_size}; k++) {
        sum += a[i*{edge_size} + k] * b[k*{edge_size} + j];
    }
    c[i*{edge_size} + j] = sum;
}
""".strip().replace("{edge_size}", str(edge_size))
matmul_shader = glsl2spirv(matmul_shader_code)
alg = mgr.algorithm([t_a, t_b, t_out], matmul_shader)
seq = mgr.sequence()
seq.record(kp.OpTensorSyncDevice([t_a, t_b, t_out]))
seq.record(kp.OpAlgoDispatch(alg))
seq.record(kp.OpTensorSyncLocal([t_out]))

# 执行计算
start_time = time.time()
for t in range(10000):
    # a = np.random.rand(edge_size * edge_size).astype(np.float32)
    # b = np.random.rand(edge_size * edge_size).astype(np.float32)
    # for i in range(edge_size * edge_size):
    #     t_a.data()[i] = a[i]
    #     t_b.data()[i] = b[i]
    seq.eval()
print("Kompute:", time.time() - start_time, "seconds")

# 验证结果
start_time = time.time()
for t in range(10000):
    # a = np.random.rand(edge_size * edge_size).astype(np.float32)
    # b = np.random.rand(edge_size * edge_size).astype(np.float32)
    np.matmul(a, b)
print("NumPy:", time.time() - start_time, "seconds")
# kp_result = np.array(t_out.data()).reshape(edge_size, edge_size)
# print("Difference:", np.linalg.norm(np_result - kp_result))
mgr.destroy()
