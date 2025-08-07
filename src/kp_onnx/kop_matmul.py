import kp
import numpy as np
from .shader_utils import compile_source


class MatMulOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        props = manager.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        local_size_x = 1
        local_size_y = 1
        while 2 * local_size_x * local_size_y <= max_workgroup_invocation:
            if 2 * local_size_x <= max_workgroup_size[0]:
                local_size_x *= 2
            if 2 * local_size_y <= max_workgroup_size[1]:
                local_size_y *= 2
            elif 2 * local_size_x > max_workgroup_size[0]:  # stop if neither x nor y can be double
                break

        self.local_size_x = local_size_x
        self.local_size_y = local_size_y
        self.sizes = [0, 0, 0]
        self.workgroup = None
        self.shader = None
        self.shader_code = """
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}) in;
layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (constant_id = 0) const float size_m_f = 0;
layout (constant_id = 1) const float size_k_f = 0;
layout (constant_id = 2) const float size_n_f = 0;

void main()
{{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint size_m = uint(size_m_f);
    uint size_k = uint(size_k_f);
    uint size_n = uint(size_n_f);
    if(row >= size_m || col >= size_n) return;
    float acc = 0.0;
    uint start_1 = row * size_k;
    for(uint i = 0, start_2 = 0; i < size_k; i++, start_2 += size_n)
        acc += in_tensor_1[start_1 + i] * in_tensor_2[start_2 + col];
    out_tensor[(row * size_n) + col] = acc;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "MatMulOp requires 2 inputs"
        assert inputs[0].ndim >= 2 and inputs[1].ndim == 2, "MatMulOp requires 2D inputs"
        rows = inputs[0].shape[0]
        if inputs[0].ndim > 2:
            for i in range(1, inputs[0].ndim - 1):
                rows *= inputs[0].shape[i]
        cols = inputs[0].shape[-1]
        ncols = inputs[1].shape[-1]
        in_1 = inputs[0].reshape(-1).astype(np.float32)
        in_2 = inputs[1].reshape(-1).astype(np.float32)
        tensor_in_1 = self.manager.tensor(in_1)
        tensor_in_2 = self.manager.tensor(in_2)
        tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))

        if self.shader is None or self.sizes != [rows, cols, ncols]:
            self.sizes = [rows, cols, ncols]
            local_size_x = min(self.local_size_x, rows)
            local_size_y = min(self.local_size_y, ncols)
            # compile shader
            self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x, local_size_y=local_size_y))
            self.workgroup = ((rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, 1)

        algo = self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                      self.shader, self.workgroup, self.sizes, [])
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        out_shape = inputs[0].shape[:-1] + (ncols,)
        outputs = [tensor_out.data().reshape(out_shape)]
        del tensor_in_1
        del tensor_in_2
        del tensor_out
        return outputs
