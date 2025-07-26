import kp
import numpy
import os
import sys
import sh_common
import subprocess
import tempfile
import time


def glsl2spirv(glsl_code):
    """
    Compile GLSL to SpirV and return as bytes.
    """

    if not isinstance(glsl_code, str):
        raise TypeError("glsl2spirv expects a string.")

    filename1 = os.path.join(tempfile.gettempdir(), "x.txt")
    filename2 = os.path.join(tempfile.gettempdir(), "x.spv")

    with open(filename1, "wb") as f:
        f.write(glsl_code.encode())

    # Note: -O means optimize, use -O0 to disable optimization
    try:
        stdout = subprocess.check_output(
            ["glslangValidator", "-S", "comp", "-V", "-Os", "-o", filename2, filename1], stderr=subprocess.STDOUT
        )
        stdout  # noqa - not used
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    with open(filename2, "rb") as f:
        binary = f.read()

    return binary


if len(sys.argv) != 3:
    print("run_vgg7.py INPUT OUTPUT")
    print(" Tiling is not implemented, but padding is implemented")
    sys.exit(1)

conv_shader_code = """
#version 450

#define index_in_no_ic(pos) (pos.x + (pos.y * uint(in_w))) * uint(in_cg)
#define index_out(pos) ((pos.x + (pos.y * uint(out_w))) * uint(out_cg)) + gl_GlobalInvocationID.z

layout (local_size_x = 8, local_size_y = 2) in;

// [y][x][group] (vec4: channels)
layout (set = 0, binding = 0) buffer buf_in_image { readonly restrict vec4 in_image[]; };
// [outputCGroups] (vec4: output channels)
layout (set = 0, binding = 1) buffer buf_in_bias { readonly restrict vec4 in_bias[]; };
// [outputCGroups][kernelH][kernelW][inputCGroups] (mat4: input & output channels)
layout (set = 0, binding = 2) buffer buf_in_weight { readonly restrict mat4 in_weight[]; };
// [y][x][group] (vec4: channels)
layout (set = 0, binding = 3) buffer buf_out_image { writeonly restrict vec4 out_image[]; };

// The 'c' measures in cgroups.
// Some maths changes as a result.
layout (constant_id = 0) const float in_w = 0;
layout (constant_id = 1) const float in_h = 0;
layout (constant_id = 2) const float in_cg = 0;
layout (constant_id = 3) const float out_w = 0;
layout (constant_id = 4) const float out_cg = 0;

void main() {
    // out x/y is gl_GlobalInvocationID.xy
    // we need to account for workgroupy padding *here*
    // so long as we aren't trying to output to a pixel that doesn't exist,
    //  we won't read from any pixels that don't exist
    if (
        (gl_GlobalInvocationID.x < (uint(in_w) - 2)) &&
        (gl_GlobalInvocationID.y < (uint(in_h) - 2))
    ) {
        vec4 value = in_bias[gl_GlobalInvocationID.z];
        for (uint x = 0; x < 3; x++) {
            for (uint y = 0; y < 3; y++) {
                uint weight_ptr = ((gl_GlobalInvocationID.z * 9) + (x + (y * 3))) * uint(in_cg);
                // specific pixel
                // important to note is that since in position has a border around it,
                // no further transformation is necessary (the - is implied)
                uvec2 in_pos = gl_GlobalInvocationID.xy + uvec2(x, y);
                uint in_ptr = index_in_no_ic(in_pos);
                for (uint icg = 0; icg < uint(in_cg); icg++) {
                    // input channel group
                    vec4 iCG = in_image[in_ptr];
                    // handle all 4 input components
                    value += iCG * in_weight[weight_ptr];
                    weight_ptr += 1;
                    in_ptr += 1;
                }
            }
        }
        // leakyrelu slope 0.1
        value = (max(value, 0.0) * 0.9) + (value * 0.1);
        out_image[index_out(gl_GlobalInvocationID.xy)] = value;
    }
}
""".strip()
conv_shader = glsl2spirv(conv_shader_code)

# NOTES:
# + Tiling is not implemented, but padding is implemented
#   So don't run anything too big through it

device_id = 0
kpm = kp.Manager(device_id)

image = sh_common.image_load(sys.argv[1])

start_time = time.time()
image = image.repeat(2, 0).repeat(2, 1)
image = numpy.pad(image, ((7, 7), (7, 7), (0, 0)), mode="edge")

# Ensure image has 4 channels even though they will be unused.
# This is because of vectorization vec4 magic.
while image.shape[2] < sh_common.VSZ:
    image = numpy.pad(image, ((0, 0), (0, 0), (0, 1)), mode="constant")

# Prepare the initial tensor.
tensor_in = kpm.tensor(image)
tensor_in_h = image.shape[0]
tensor_in_w = image.shape[1]
tensor_in_cg = 1
tensor_in_c = 3

# Run things.
channels = [32, 32, 64, 64, 128, 128, 3]

for i in range(7):
    # Prepare tensors.
    # 'c' is the total amount of channels, while 'cg' is the amount of vec4s (channel-groups).
    # This is important because weights have to be padded for the shader.
    tensor_out_h = tensor_in_h - 2
    tensor_out_w = tensor_in_w - 2
    tensor_out_c = channels[i]
    tensor_out_cg = (channels[i] + (sh_common.VSZ - 1)) // sh_common.VSZ
    # TODO: How to produce a blank tensor we don't care about the contents of?
    # This isn't being synced, and from experience so far that should handle most of it,
    #  but what about memory usage?
    # *Most* of these tensors live entirely on-device except when debugging.
    # Can that be handled? (Also good question: Does it even need to be handled?)
    tensor_out = kpm.tensor(numpy.zeros((tensor_out_h * tensor_out_w * tensor_out_cg * sh_common.VSZ)))
    weight = kpm.tensor(sh_common.load_weights_padded("kipper", (i * 2) + 0, tensor_out_c, tensor_in_c, 3))
    bias = kpm.tensor(sh_common.load_biases_padded("kipper", (i * 2) + 1, tensor_out_c))
    # Compute.
    # TODO: It'd be nice to wrap this up into a class for optimization purposes.
    workgroup = ((tensor_out_w + 7) // 8, (tensor_out_h + 1) // 2, tensor_out_cg)
    alg = kpm.algorithm(
        # tensors
        [tensor_in, bias, weight, tensor_out],
        # spirv
        conv_shader,
        # workgroup
        workgroup,
        # spec_consts
        [tensor_in_w, tensor_in_h, tensor_in_cg, tensor_out_w, tensor_out_cg],
        # push_consts
        []
    )

    print("Step complexity " + str(workgroup))
    print("Step channel layout " + str(tensor_in_cg) + " " + str(tensor_out_cg))

    # Do this first. Keep in mind "syncs" are copies.
    last_seq = kpm.sequence()
    things_to_sync_to_device = [bias, weight]
    if i == 0:
        # For first layer, the input isn't on-device yet
        things_to_sync_to_device.append(tensor_in)
    last_seq.eval_async(kp.OpTensorSyncDevice(things_to_sync_to_device))
    last_seq.eval_await()

    # Prepare
    seq = (kpm.sequence()
           .record(kp.OpAlgoDispatch(alg, []))
           )
    # Run
    seq.eval()

    print(f"Done with step {i+1}")

    # Swap over.
    tensor_in = tensor_out
    tensor_in_h = tensor_out_h
    tensor_in_w = tensor_out_w
    tensor_in_c = tensor_out_c
    tensor_in_cg = tensor_out_cg

# Download output
fin_seq = kpm.sequence()
fin_seq.eval_async(kp.OpTensorSyncLocal([tensor_in]))
fin_seq.eval_await()

print(kpm.list_devices()[device_id]["device_name"], "used", time.time() - start_time, "seconds")

# Output
out_na = tensor_in.data().reshape((tensor_in_h, tensor_in_w, tensor_in_cg * sh_common.VSZ))
# Crop off 'alpha'
out_na = out_na[:, :, 0:3]
sh_common.image_save(sys.argv[2], out_na)
kpm.destroy()
