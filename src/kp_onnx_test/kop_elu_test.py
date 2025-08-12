from kp import Manager
import numpy as np
from time import perf_counter
from src.kp_onnx.kop_elu import EluOp

# Device info
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

# Op
elu_op = EluOp(mgr, ['data', 'alpha'], ['output'])

# ---------------- Case 1: default alpha=1.0, 1D ----------------
print("Case 1: ELU alpha=1.0 (default)")
x = (np.random.random(1024 * 1024).astype(np.float32) - 0.5) * 8.0
print("Input shape:", x.shape)

t0 = perf_counter()
np_out = np.where(x > 0, x, np.exp(x) - 1.0).astype(np.float32)
t1 = perf_counter()
print("Numpy: ", t1 - t0, "seconds")

t0 = perf_counter()
kp_out = elu_op.run(x)[0]
t1 = perf_counter()
print(f"{elu_op}: ", t1 - t0, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: alpha=0.5, 2D ----------------
print("Case 2: ELU alpha=0.5, 2D")
x = (np.random.random((512, 1024)).astype(np.float32) - 0.5) * 6.0
alpha = 0.5
print("Input shape:", x.shape)

t0 = perf_counter()
np_out = np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
t1 = perf_counter()
print("Numpy: ", t1 - t0, "seconds")

t0 = perf_counter()
kp_out = elu_op.run(x, alpha)[0]
t1 = perf_counter()
print(f"{elu_op}: ", t1 - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 3: alpha=1.5, non-square ----------------
print("Case 3: ELU alpha=1.5, non-square")
x = (np.random.random((300, 777)).astype(np.float32) - 0.5) * 10.0
alpha = 1.5
print("Input shape:", x.shape)

t0 = perf_counter()
np_out = np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
t1 = perf_counter()
print("Numpy: ", t1 - t0, "seconds")

t0 = perf_counter()
kp_out = elu_op.run(x, alpha)[0]
t1 = perf_counter()
print(f"{elu_op}: ", t1 - t0, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 4: small input sanity ----------------
print("Case 4: small sanity")
x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
alpha = 1.0
print("Input shape:", x.shape)

t0 = perf_counter()
np_out = np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
t1 = perf_counter()
print("Numpy: ", t1 - t0, "seconds")

t0 = perf_counter()
kp_out = elu_op.run(x, alpha)[0]
t1 = perf_counter()
print(f"{elu_op}: ", t1 - t0, "seconds")

# print("Input:    ", x)
# print("Numpy ELU:", np_out)
# print("Kp   ELU :", kp_out)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
