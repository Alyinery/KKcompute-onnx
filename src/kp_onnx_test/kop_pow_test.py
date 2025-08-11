from kp import Manager
import numpy as np
import time
from kp_onnx.kop_pow import PowOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

pow_op = PowOp(mgr, ['input_base', 'input_exp'], ['output'])
numpy_in = np.random.random(1024 * 1024 * 16)
numpy_exp = np.array(2.0)

start_time = time.time()
numpy_out = np.power(numpy_in, numpy_exp)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = pow_op.run(numpy_in, numpy_exp)[0]
print(f"{pow_op}:", time.time() - start_time, "seconds")

print(numpy_out)
print(kp_out)
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

