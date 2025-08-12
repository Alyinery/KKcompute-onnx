import numpy as np
import time
from kp import Manager
from src.kp_onnx.kop_sub import SubOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sub_op = SubOp(mgr, ['input_1', 'input_2'], ['output'])

input_1 = np.random.rand(1024 * 1024).astype(np.float32).reshape(1024, 1024)
input_2 = np.random.rand(1024 * 1024).astype(np.float32).reshape(1024, 1024)

start_time = time.time()
numpy_out = np.subtract(input_1, input_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sub_op.run(input_1, input_2)[0]
print(f"{sub_op}: ", time.time() - start_time, "seconds")

# print(numpy_out)
# print(kp_out)
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
