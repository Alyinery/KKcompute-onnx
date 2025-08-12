from kp import Manager
import numpy as np
import time
from src.kp_onnx.kop_gemm import GemmOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

gemm_op = GemmOp(mgr, ['A', 'B', 'C'], ['Y'])

# ---- Case 1: basic A @ B (no C) ----
print("Case 1: basic A @ B (no C)")
A = (np.random.random((128, 64)).astype(np.float32) - 0.5) * 2.0
B = (np.random.random((64, 96)).astype(np.float32) - 0.5) * 2.0

print("A shape:", A.shape)
print("B shape:", B.shape)

start_time = time.time()
np_out = np.dot(A, B)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 2: with C (M,N), alpha/beta ----
print("Case 2: with C (M,N), alpha/beta")
M, K, N = 64, 128, 32
A = (np.random.random((M, K)).astype(np.float32) - 0.5) * 2.0
B = (np.random.random((K, N)).astype(np.float32) - 0.5) * 2.0
C = (np.random.random((M, N)).astype(np.float32) - 0.5) * 2.0
alpha, beta = 0.75, 0.5

start_time = time.time()
np_out = alpha * (np.dot(A, B)) + beta * C
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gemm_op.run(A, B, C, alpha, beta)[0]
print(f"{gemm_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 3: C scalar ----
print("Case 3: C scalar")
A = (np.random.random((32, 48)).astype(np.float32) - 0.5) * 2.0
B = (np.random.random((48, 24)).astype(np.float32) - 0.5) * 2.0
C_scalar = 0.1
alpha, beta = 1.2, 0.3

start_time = time.time()
np_out = alpha * (np.dot(A, B)) + beta * C_scalar
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gemm_op.run(A, B, C_scalar, alpha, beta)[0]
print(f"{gemm_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 4: C broadcast as (1,N) ----
print("Case 4: C broadcast as (1,N)")
A = (np.random.random((40, 25)).astype(np.float32) - 0.5) * 2.0
B = (np.random.random((25, 30)).astype(np.float32) - 0.5) * 2.0
C_row = (np.random.random((1, 30)).astype(np.float32) - 0.5) * 2.0
alpha, beta = 0.9, 0.6

start_time = time.time()
np_out = alpha * (np.dot(A, B)) + beta * np.broadcast_to(C_row, (40, 30))
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gemm_op.run(A, B, C_row, alpha, beta)[0]
print(f"{gemm_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# ---- Case 5: transA / transB ----
print("Case 5: transA / transB")
A = (np.random.random((30, 40)).astype(np.float32) - 0.5) * 2.0  # 30x40
B = (np.random.random((25, 30)).astype(np.float32) - 0.5) * 2.0  # 25x30
# Use A^T(40x30) @ B^T(30x25) = (40x25)
alpha, beta = 1.0, 0.0

start_time = time.time()
np_out = np.dot(A.T, B.T)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gemm_op.run(A, B, None, alpha, beta, 1, 1)[0]  # transA=1, transB=1
print(f"{gemm_op}:", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
