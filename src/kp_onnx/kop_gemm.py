import numpy as np
import kp
from .kop_matmul import MatMulOp
from .shader_utils import compile_source


class GemmOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.matmul = MatMulOp(manager, ['A', 'B'], ['P'])

        self.local_size_x = None
        self.shader = None
        self.shader_tmpl = """
#version 450
layout(local_size_x = {local_size_x}, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer PBuf   {{ float P[];  }};
layout(set=0, binding=1) buffer CBuf   {{ float C[];  }};
layout(set=0, binding=2) buffer YBuf   {{ float Y[];  }};
layout(set=0, binding=3) buffer Sizes  {{ int   S[];  }}; // [MN]
layout(set=0, binding=4) buffer Scalars{{ float K[];  }}; // [alpha, beta]

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    int MN = S[0];
    if (gid >= uint(MN)) return;
    float y = K[0] * P[gid] + K[1] * C[gid];
    Y[gid] = y;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GemmOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # Support：run(A,B[,C[, alpha[, beta[, transA[, transB]]]]])
        if len(inputs) < 2:
            raise AssertionError("GemmOp expects at least A and B")

        A, B = np.asarray(inputs[0], np.float32), np.asarray(inputs[1], np.float32)
        C = None
        alpha, beta = 1.0, 1.0
        transA = 0
        transB = 0

        idx = 2
        if len(inputs) > idx:
            C = inputs[idx]; idx += 1
        if len(inputs) > idx and inputs[idx] is not None:
            alpha = float(inputs[idx]); idx += 1
        if len(inputs) > idx and inputs[idx] is not None:
            beta  = float(inputs[idx]); idx += 1
        if len(inputs) > idx and inputs[idx] is not None:
            transA = int(inputs[idx]); idx += 1
        if len(inputs) > idx and inputs[idx] is not None:
            transB = int(inputs[idx]); idx += 1

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"A and B must be 2-D, got {A.shape} and {B.shape}")

        if transA:
            A = np.ascontiguousarray(A.T, dtype=np.float32)
        if transB:
            B = np.ascontiguousarray(B.T, dtype=np.float32)

        M, K1 = A.shape
        K2, N = B.shape
        if K1 != K2:
            raise ValueError(f"Incompatible inner dims after transpose: A.K={K1}, B.K={K2}")
        MN = M * N

        # 1) P = A @ B
        P = self.matmul.run(A, B)[0]   # shape (M, N)

        # 2) C broadcasting to (M,N)
        if C is None:
            Cb = np.zeros((M, N), dtype=np.float32)
        else:
            Cb = np.asarray(C, dtype=np.float32)
            try:
                Cb = np.broadcast_to(Cb, (M, N)).astype(np.float32, copy=False)
            except Exception as e:
                raise ValueError(f"C is not broadcastable to (M={M}, N={N}); got {np.asarray(C).shape}") from e

        props = self.manager.get_device_properties()
        max_inv = int(props["max_work_group_invocations"])
        max_x = int(props["max_work_group_size"][0])
        lx_cap = max(1, min(max_inv, max_x, MN))
        want_lx = 1
        while (want_lx << 1) <= lx_cap:
            want_lx <<= 1

        if (self.shader is None) or (self.local_size_x != want_lx):
            self.local_size_x = want_lx
            self.shader = compile_source(self.shader_tmpl.format(local_size_x=want_lx))

        tensor_P = self.manager.tensor(P.reshape(-1).astype(np.float32, copy=False))
        tensor_C = self.manager.tensor(Cb.reshape(-1).astype(np.float32, copy=False))
        tensor_Y = self.manager.tensor(np.empty(MN, dtype=np.float32))
        tensor_S = self.manager.tensor(np.asarray([MN], dtype=np.int32))
        tensor_K = self.manager.tensor(np.asarray([alpha, beta], dtype=np.float32))

        groups_x = (MN + self.local_size_x - 1) // self.local_size_x
        workgroup = (int(groups_x), 1, 1)

        algo = self.manager.algorithm([tensor_P, tensor_C, tensor_Y, tensor_S, tensor_K], self.shader, workgroup)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_P, tensor_C, tensor_S, tensor_K])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_Y])) \
           .eval()

        output = [tensor_Y.data().reshape(M, N)]

        del tensor_P, tensor_C, tensor_Y, tensor_S, tensor_K, algo, seq
        return output
