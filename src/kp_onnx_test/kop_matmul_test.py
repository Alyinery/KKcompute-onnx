from kp import Manager
import numpy as np
import time
from kp_onnx.kop_matmul import MatMulOp

device_id = 1
mgr = Manager(device_id)
print(mgr.list_devices()[device_id])

matmul_op = MatMulOp(mgr, ['input1', 'input2'], ['output'])
numpy_in_1 = np.random.random((5, 1000, 2000))
numpy_in_2 = np.random.random((2000, 500))

start_time = time.time()
numpy_out = np.matmul(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

# matmul_op.run(numpy_in_1, numpy_in_2)  # warm up
start_time = time.time()
kp_out = matmul_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_op}:", time.time() - start_time, "seconds")

print(numpy_out)
print(kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
