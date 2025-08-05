import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import pow


@python2shader
def compute_shader_pow(index=("input", "GlobalInvocationId", ivec2),
                       in_base=("buffer", 0, Array(f32)),
                       in_exp=("buffer", 1, Array(f32)),
                       out_pow=("buffer", 2, Array(f32))):
    i = index.x
    out_pow[i] = pow(in_base[i], in_exp[0])


_pow_code = compute_shader_pow.to_spirv()


class PowOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PowOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PowOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "PowOp requires two inputs"
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in_base = self.manager.tensor(numpy_in)
        tensor_in_exp = self.manager.tensor(inputs[1])
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in_base, tensor_in_exp, tensor_out], _pow_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in_base, tensor_in_exp])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_in_base
        del tensor_in_exp
        del tensor_out
        return outputs
