import numpy as np
import time
from kp import Manager
from src.kp_onnx.kop_log import LogOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

log_op = LogOp(mgr, ['input'], ['output'])

# 保证输入为正，避免 log(x <= 0)
x = np.random.rand(1024 * 1024).astype(np.float32).reshape(1024, 1024) + 1e-2

start_time = time.time()
numpy_out = np.log(x)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kop_out = log_op.run(x)[0]
print("Kompute:", time.time() - start_time, "seconds")

# print(numpy_out)
# print(kop_out)
print("Max error:", np.max(np.abs(numpy_out - kop_out)))
print(np.allclose(numpy_out, kop_out, rtol=1e-4, atol=1e-4))
