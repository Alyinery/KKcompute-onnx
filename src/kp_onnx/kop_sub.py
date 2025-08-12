import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array


@python2shader
def compute_shader_sub(index=("input", "GlobalInvocationId", ivec2),
                       in_data1=("buffer", 0, Array(f32)),
                       in_data2=("buffer", 1, Array(f32)),
                       out_data=("buffer", 2, Array(f32))):
    i = index.x
    out_data[i] = in_data1[i] - in_data2[i]


_sub_code = compute_shader_sub.to_spirv()


class SubOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SubOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SubOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "SubOp requires 2 inputs"
        assert inputs[0].shape == inputs[1].shape, "The inputs must have the same shape"
        tensor_shape = inputs[0].shape
        in1 = inputs[0].reshape(-1).astype(np.float32)
        in2 = inputs[1].reshape(-1).astype(np.float32)
        tensor_in1 = self.manager.tensor(in1)
        tensor_in2 = self.manager.tensor(in2)
        tensor_out = self.manager.tensor(np.zeros_like(in1))
        algo = self.manager.algorithm([tensor_in1, tensor_in2, tensor_out], _sub_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in1, tensor_in2])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_in1, tensor_in2, tensor_out
        return outputs
