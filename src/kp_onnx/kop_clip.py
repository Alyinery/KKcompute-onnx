import numpy as np
import kp
from .shader_utils import compile_source


class ClipOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.local_size_x = None
        self.shader = None

        self.shader_code = """
#version 450
layout(local_size_x = {local_size_x}, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf  {{ float in_data[];  }};
layout(set=0, binding=1) buffer OutBuf {{ float out_data[]; }};

// 纯数组 SSBO，避免 struct+bool 的跨驱动对齐差异
layout(set=0, binding=2) buffer Scalars {{ float scalars[]; }};  // [scalar_min, scalar_max]
layout(set=0, binding=3) buffer MinArr  {{ float min_array[];  }};
layout(set=0, binding=4) buffer MaxArr  {{ float max_array[];  }};
layout(set=0, binding=5) buffer IntsBuf {{ int   ints[];     }}; // [size, use_s_min, use_s_max, has_min, has_max]

void main() {{
    uint gid = gl_GlobalInvocationID.x;

    int size      = ints[0];
    int use_s_min = ints[1];
    int use_s_max = ints[2];
    int has_min   = ints[3];
    int has_max   = ints[4];

    if (gid >= uint(size)) {{ return; }}

    float v = in_data[gid];

    if (has_min != 0) {{
        float m = (use_s_min != 0) ? scalars[0] : min_array[gid];
        if (v < m) v = m;
    }}
    if (has_max != 0) {{
        float M = (use_s_max != 0) ? scalars[1] : max_array[gid];
        if (v > M) v = M;
    }}

    out_data[gid] = v;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ClipOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        if len(inputs) < 1:
            raise ValueError("ClipOp need at least one input")

        data = np.asarray(inputs[0], dtype=np.float32)
        flat_data = data.reshape(-1).astype(np.float32)
        data_size = int(flat_data.size)

        # ---- parse min/max ----
        min_param = inputs[1] if len(inputs) > 1 else None
        max_param = inputs[2] if len(inputs) > 2 else None

        has_min = min_param is not None
        has_max = max_param is not None

        use_scalar_min = False
        use_scalar_max = False
        scalar_min = 0.0
        scalar_max = 0.0

        min_array = np.zeros(1, dtype=np.float32)
        max_array = np.zeros(1, dtype=np.float32)

        if has_min:
            if np.isscalar(min_param):
                use_scalar_min = True
                scalar_min = float(min_param)
            else:
                arr = np.asarray(min_param, dtype=np.float32).reshape(-1)
                if arr.size != data_size:
                    raise ValueError(f"Min array size must match input data size, expected {data_size}, actual {arr.size}")
                min_array = arr

        if has_max:
            if np.isscalar(max_param):
                use_scalar_max = True
                scalar_max = float(max_param)
            else:
                arr = np.asarray(max_param, dtype=np.float32).reshape(-1)
                if arr.size != data_size:
                    raise ValueError(f"Max array size must match input data size, expected {data_size}, actual {arr.size}")
                max_array = arr

        props = self.manager.get_device_properties()
        max_inv = int(props["max_work_group_invocations"])
        max_x = int(props["max_work_group_size"][0])
        lx_cap = max(1, min(max_inv, max_x, data_size))
        want_lx = 1
        while (want_lx << 1) <= lx_cap:
            want_lx <<= 1

        if (self.shader is None) or (self.local_size_x != want_lx):
            self.local_size_x = want_lx
            self.shader = compile_source(self.shader_code.format(local_size_x=self.local_size_x))

        tensor_in = self.manager.tensor(flat_data)
        tensor_out = self.manager.tensor(np.empty_like(flat_data))

        scalars = np.asarray([scalar_min, scalar_max], dtype=np.float32)
        tensor_scalars = self.manager.tensor(scalars)

        tensor_min_array = self.manager.tensor(min_array)
        tensor_max_array = self.manager.tensor(max_array)

        ints = np.asarray([
            data_size,
            int(use_scalar_min),
            int(use_scalar_max),
            int(has_min),
            int(has_max),
        ], dtype=np.int32)
        tensor_ints = self.manager.tensor(ints)

        tensors = [
            tensor_in,         # binding 0
            tensor_out,        # binding 1
            tensor_scalars,    # binding 2
            tensor_min_array,  # binding 3
            tensor_max_array,  # binding 4
            tensor_ints,       # binding 5
        ]

        groups_x = (data_size + self.local_size_x - 1) // self.local_size_x
        workgroup = (int(groups_x), 1, 1)

        algo = self.manager.algorithm(
            tensors,
            self.shader,
            workgroup,
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice(tensors)) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(data.shape)]

        del tensor_in, tensor_out, tensor_scalars, tensor_min_array, tensor_max_array, tensor_ints, algo, seq
        return outputs
