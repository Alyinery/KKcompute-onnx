import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import exp, sign, abs


@python2shader
def compute_shader_erf(index=("input", "GlobalInvocationId", ivec2),
                       data1=("buffer", 0, Array(f32)),
                       data2=("buffer", 1, Array(f32))):
    i = index.x
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    x = data1[i]
    s = sign(x)
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs * x_abs)
    data2[i] = s * y


_erf_code = compute_shader_erf.to_spirv()


class ErfOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ErfOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ErfOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in, tensor_out], _erf_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        seq.destroy()
        algo.destroy()
        tensor_in.destroy()
        tensor_out.destroy()
        return outputs

