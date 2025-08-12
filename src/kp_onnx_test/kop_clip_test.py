from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_clip import ClipOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

clip_op = ClipOp(mgr, ['data', 'min', 'max'], ['output'])

# ---- Case 1: min: None, max: None ----
print("Case 1: min: None, max: None")
x = (np.random.random(1024 * 1024).astype(np.float32) - 0.5) * 8.0

# NumPy
start_time = time.time()
np_out = x.copy()
print(f"Numpy: ", time.time() - start_time, "seconds")

# Vulkan
start_time = time.time()
kp_out = clip_op.run(x)[0]

print(f"{clip_op}: ", time.time() - start_time, "seconds")

# Validate the results
print(f"Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 2: min: Scalar, max: None ----
print("Case 2: min: Scalar, max: None")
x = (np.random.random(1024 * 1024).astype(np.float32) - 0.5) * 8.0
min_val = 0.0

# NumPy
start_time = time.time()
np_out = np.maximum(x, min_val)
print(f"Numpy: ", time.time() - start_time, "seconds")

# Vulkan
start_time = time.time()
kp_out = clip_op.run(x, min_val)[0]
print(f"{clip_op}: ", time.time() - start_time, "seconds")

# Validate the results
print(f"Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 3: min: None, max: Scalar ----
print("Case 3: min: None, max: Scalar")
x = (np.random.random(1024 * 1024).astype(np.float32) - 0.5) * 8.0
max_val = 1.25

# NumPy
start_time = time.time()
np_out = np.clip(x, None, max_val)
print(f"Numpy: ", time.time() - start_time, "seconds")

# Vulkan
start_time = time.time()
kp_out = clip_op.run(x, None, max_val)[0]
print(f"{clip_op}: ", time.time() - start_time, "seconds")

# Validate the results
print(f"Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 4: min: Scalar max: Scalar ----
print("Case 4: min: Scalar, max: Scalar")
x = (np.random.random(1024 * 1024).astype(np.float32) - 0.5) * 8.0
min_val = 0.0
max_val = 6.0

# NumPy
start_time = time.time()
np_out = np.clip(x, min_val, max_val)
print(f"Numpy: ", time.time() - start_time, "seconds")

# Vulkan
start_time = time.time()
kp_out = clip_op.run(x, min_val, max_val)[0]
print(f"{clip_op}: ", time.time() - start_time, "seconds")

# Validate the results
print(f"Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 5: min: array, max: array ----
print("Case 5: min: array, max: array")
x = (np.random.random((512, 1024)).astype(np.float32) - 0.5) * 8.0
N = x.size
min_arr = (np.random.random(N).astype(np.float32) * -2.0).reshape(x.shape)
max_arr = (np.random.random(N).astype(np.float32) * 2.0 + 0.5).reshape(x.shape)

# Ensure min <= max
pair_min = np.minimum(min_arr, max_arr)
pair_max = np.maximum(min_arr, max_arr)

# NumPy
start_time = time.time()
np_out = np.clip(x, pair_min, pair_max)
print(f"Numpy: ", time.time() - start_time, "seconds")

# Vulkan
start = time.time()
kp_out = clip_op.run(x, pair_min, pair_max)[0]
kp_time = time.time() - start
print(f"{clip_op}: ", time.time() - start_time, "seconds")

# Validate the results
print(f"Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


