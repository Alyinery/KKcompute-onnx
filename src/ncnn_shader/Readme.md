# NCNN GLSL extension

## rationale
Different GPUs support different features, some support fp16 as buffer storage type, some support fp16 as operand variable, some old GPUs only support fp32

When the GPU supports the `VK_KHR_16bit_storage` extension, in order to minimize the memory bandwidth consumption of the GPU, we will give priority to using fp16 as the storage type. Otherwise, we use `packHalf2x16` and `unpackHalf2x16` in GLSL 4.2 to compress 2 fp32 to uint, reducing read and write bandwidth.

Similarly, when the gpu supports the `VK_KHR_shader_float16_int8` extension, in order to speed up the calculation efficiency, we will give priority to using fp16 as the operation operand, which usually doubles the speed. Otherwise, we use fp32.

To ensure the widest compatibility, the following code for declaring descriptor binding and loading data will be written

```c
#if NCNN_fp16_storage // gpu supports 16bit storage
layout (binding = 0) buffer blob { f16vec4 blob_data[]; };
#elif NCNN_fp16_packed // gpu supports GLSL 4.2
layout (binding = 0) buffer blob { uvec2 blob_data[]; };
#else // gpu only supports fp32
layout (binding = 0) buffer blob { vec4 blob_data[]; };
#endif

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

#if NCNN_fp16_storage && NCNN_fp16_arithmetic // gpu supports 16bit storage and shader float16
    f16vec4 x = blob_data[i];
#elif NCNN_fp16_storage // gpu supports 16bit storage but no shader float16
    vec4 x = vec4(blob_data[i]);
#elif NCNN_fp16_packed && NCNN_fp16_arithmetic // gpu supports GLSL 4.2 and shader float16
    f16vec4 x = f16vec4(unpackFloat2x16(blob_data[i].x), unpackFloat2x16(blob_data[i].y));
#elif NCNN_fp16_packed // gpu supports GLSL 4.2
    vec4 x = vec4(unpackHalf2x16(blob_data[i].x), unpackHalf2x16(blob_data[i].y));
#else // gpu only supports fp32
    vec4 x = blob_data[i];
#endif
}
```

As you can see, just declaring the buffer type and reading a value consumes a lot of lines of code, which is a maintenance nightmare. Therefore, ncnn adds more flexible data types and auxiliary functions to reduce the size of the code and improve readability, and will automatically expand to the most efficient implementation according to the feature level supported by the GPU.

The above code, by using the ncnn glsl extension, can be simplified to

```c
layout (binding = 0) buffer blob { sfpvec4 blob_data[]; };

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

    afpvec4 x = buffer_ld4(blob_data, i);
}
```

The ncnn glsl extension beyond the OpenGL® Shading Language (version 4.50) provides the necessary data types for storage, computation, shared memory, and load, store, conversion functions for buffers and images. We also provide some buffer and image copy functions to prevent loss of precision when using fp16 as the intermediate data type, and to avoid unnecessary `unpackHalf2x16` and `packHalf2x16` pair.

# data types

## storage type

declare buffer data layout in descriptor binding

```c
layout (binding = 0) buffer top_blob { sfpvec4 top_blob_data[]; };
```

|storage type|fp32|fp16p|fp16s|
|---|---|---|---|
|sfp|float|uint|float16_t|
|sfpvec2|vec2|uint|f16vec2|
|sfpvec4|vec4|uvec2|f16vec4|
|sfpvec8|mat2x4|uvec4|f16mat2x4|

## arithmetic type

declare local variable in glsl code

```c
void main()
{
    afpvec4 v = a * b;
}
```

|arithmetic type|fp32|fp16a|
|---|---|---|
|afp|float|float16_t|
|afpvec2|vec2|f16vec2|
|afpvec4|vec4|f16vec4|
|afpvec8|mat2x4|f16mat2x4|

## local type

declare variable in shared local memory

```c
shared lfp tmp_a[8][4][2];
```

|local type|fp32|fp16p / fp16s only|fp16s+fp16a|fp16s+fp16u|
|---|---|---|---|---|
|lfp|float|float|float|float16_t|
|lfpvec4|vec4|uvec2|uint64_t|f16vec4|

# buffer functions

- load typed value from src[offset]

```c
afp buffer_ld1(sfp src, int offset);
afpvec2 buffer_ld2(sfpvec2 src, int offset);
afpvec4 buffer_ld4(sfpvec4 src, int offset);
afpvec8 buffer_ld8(sfpvec8 src, int offset);
```

- store typed value to dst[offset]

```c
void buffer_st1(sfp dst, int offset, afp v);
void buffer_st2(sfpvec2 dst, int offset, afpvec2 v);
void buffer_st4(sfpvec4 dst, int offset, afpvec4 v);
void buffer_st8(sfpvec8 dst, int offset, afpvec8 v);
```

- copy typed value from src[src_offset] to dst[dst_offset]

```c
void buffer_cp1(sfp dst, int dst_offset, sfp src, int src_offset);
void buffer_cp2(sfpvec2 dst, int dst_offset, sfpvec2 src, int src_offset);
void buffer_cp4(sfpvec4 dst, int dst_offset, sfpvec4 src, int src_offset);
void buffer_cp8(sfpvec4 dst, int dst_offset, sfpvec4 src, int src_offset);
```

- copy and pack value from src[src_offsets[0],src_offsets[1],...] to dst[dst_offset]

```c
void buffer_cp1to4(sfpvec4 dst, int dst_offset, sfp src, ivec4 src_offsets);
void buffer_cp1to8(sfpvec8 dst, int dst_offset, sfp src, ivec4 src_offsets_0, ivec4 src_offsets_1);
void buffer_cp4to8(sfpvec8 dst, int dst_offset, sfpvec4 src, ivec2 src_offsets);
```

- copy and unpack value from src[src_offset] to dst[dst_offsets[0],dst_offsets[1],...]

```c
void buffer_cp4to1(sfp dst, ivec4 dst_offsets, sfpvec4 src, int src_offset);
void buffer_cp8to1(sfp dst, ivec4 dst_offsets_0, ivec4 dst_offsets_1, sfpvec8 src, int src_offset);
void buffer_cp8to4(sfpvec4 dst, ivec2 dst_offsets, sfpvec8 src, int src_offset);
```
# local data conversion functions

- storage buffer to local memory

```c
lfp sfp2lfp(sfp v);
lfpvec4 sfp2lfpvec4(sfpvec4 v);
```

- local memory to local variable

```c
afp lfp2afp(lfp v);
afpvec4 lfp2afpvec4(lfpvec4 v);
```

Note: The common usage of local memory is to read from global memory first, store it in local memory, and then read local variables from local memory for subsequent use. Therefore, only storage type to local type and local type to arithmetic type conversion functions are provided here.

# misc functions

- prefer specialization constant over push constant

```c
T psc(T x)
```

Declare the same variable in specialization constant AND push constant section, then `psc(x)` will become a compile-time constant when specialization constant given non-zero or be dynamic via push constant otherwise. This is often used for tensor shape specialization. We can usually resolve all shape information and make them be compile-time constants for more aggressive shader optimization.

```c
layout (constant_id = 0) const int size = 0;

layout (push_constant) uniform parameter
{
    int size;
} p;

void main()
{
    const int s = psc(size);
}
```
# implementation

|option|meaning|
|---|---|
|opt.use_fp16_storage|the device has fp16 storage support|
|opt.use_fp16_arithmetic|the device has fp16 arithmetic support|
|opt.use_fp16_uniform|the device has fp16 uniform buffer support|
|opt.use_fp16_packed|the device has fp16 packed storage support for GLSL 4.2|
|opt.use_int8_storage|the device has int8 storage support|
|opt.use_int8_arithmetic|the device has int8 arithmetic support|
|opt.use_int8_uniform|the device has int8 uniform buffer support|
|opt.use_int8_packed|the device has int8 packed storage support for GLSL 4.2|
|opt.use_shader_local_memory|enable local memory optimization on discrete gpu|
|opt.use_subgroup_ops|the device has subgroup operation support|
|opt.use_cooperative_matrix|the device has cooperative matrix support|

```c
int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv)
{
    DefinitionCollector custom_defines;
    DefinitionCollector device_defines;

    if (opt.use_fp16_storage)
    {
        custom_defines.append("sfp", "float16_t");
        custom_defines.append("sfpvec2", "f16vec2");
        custom_defines.append("sfpvec4", "f16vec4");

        if (opt.use_fp16_arithmetic)
        {
            custom_defines.append("sfpvec8", "f16mat2x4");
            custom_defines.append("sfpmat4", "f16mat4");
        }
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("sfp", "uint");
        custom_defines.append("sfpvec2", "uint");
        custom_defines.append("sfpvec4", "uvec2");
        custom_defines.append("sfpvec8", "uvec4");
    }
    else
    {
        custom_defines.append("sfp", "float");
        custom_defines.append("sfpvec2", "vec2");
        custom_defines.append("sfpvec4", "vec4");
        custom_defines.append("sfpvec8", "mat2x4");
        custom_defines.append("sfpmat4", "mat4");
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.append("afp", "float16_t");
        custom_defines.append("afpvec2", "f16vec2");
        custom_defines.append("afpvec4", "f16vec4");
        custom_defines.append("afpvec8", "f16mat2x4");
        custom_defines.append("afpmat4", "f16mat4");
    }
    else
    {
        custom_defines.append("afp", "float");
        custom_defines.append("afpvec2", "vec2");
        custom_defines.append("afpvec4", "vec4");
        custom_defines.append("afpvec8", "mat2x4");
        custom_defines.append("afpmat4", "mat4");
    }

    if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
    {
        custom_defines.append("lfp", "float16_t");
        custom_defines.append("lfpvec4", "f16vec4");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "uint64_t");
    }
    else if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "uvec2");
    }
    else
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "vec4");
    }

    if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "float(v)");
        custom_defines.append("sfp2lfpvec4(v)", "pack64(halfBitsToUInt16(v))");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("lfp2afpvec4(v)", "int16BitsToHalf(unpack16(v))");
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("lfp2afpvec4(v)", "f16vec4(unpackFloat2x16(v.x),unpackFloat2x16(v.y))");
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.append("sfp2lfp(v)", "float(v)");
        custom_defines.append("sfp2lfpvec4(v)", "uvec2(packHalf2x16(vec4(v).rg),packHalf2x16(vec4(v).ba))");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
    }
    else
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
    }

    if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=f16vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}");
        custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=f16mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}");
        custom_defines.append("buffer_ld2(buf,i)", "buf[i]");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "buf[i]");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}");
        custom_defines.append("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=f16mat2x4(sbuf[si2.r],sbuf[si2.g]);}");
        custom_defines.append("buffer_ld8(buf,i)", "buf[i]");
        custom_defines.append("buffer_st8(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{f16mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}");
        custom_defines.append("buffer_cp8to4(buf,i2,sbuf,si)", "{f16mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}");
        custom_defines.append("sfp2afpmat4(v)", "v");
        custom_defines.append("afp2sfpmat4(v)", "v");
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        // custom_defines.append("buffer_ld1(buf,i)", "float16_t(buf[i])");
        custom_defines.append("buffer_ld1(buf,i)", "float16_t(unpackHalf2x16(buf[(i)/2])[(i)%2])");
        // custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=float(v);}");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        // custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        // custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packFloat2x16(f16vec2(sbuf[si4.r],sbuf[si4.g])),packFloat2x16(f16vec2(sbuf[si4.b],sbuf[si4.a])));}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        // custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=uvec4(packFloat2x16(f16vec2(sbuf[si4.r],sbuf[si4.g])),packFloat2x16(f16vec2(sbuf[si4.b],sbuf[si4.a])),packFloat2x16(f16vec2(sbuf[sii4.r],sbuf[sii4.g])),packFloat2x16(f16vec2(sbuf[sii4.b],sbuf[sii4.a])));}");

        custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _sii4d2=uvec4(sii4)/2;uvec4 _si4m2=uvec4(si4)%2;uvec4 _sii4m2=uvec4(sii4)%2; buf[i]=uvec4(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_sii4d2.r])[_sii4m2.r],unpackHalf2x16(sbuf[_sii4d2.g])[_sii4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_sii4d2.b])[_sii4m2.b],unpackHalf2x16(sbuf[_sii4d2.a])[_sii4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackFloat2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packFloat2x16(v)}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "f16vec4(unpackFloat2x16(buf[i].x),unpackFloat2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packFloat2x16(v.rg),packFloat2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; f16vec2 _v0=unpackFloat2x16(_v.x);f16vec2 _v1=unpackFloat2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");

        custom_defines.append("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}");
        custom_defines.append("buffer_ld8(buf,i)", "f16mat2x4(f16vec4(unpackFloat2x16(buf[i].r),unpackFloat2x16(buf[i].g)),f16vec4(unpackFloat2x16(buf[i].b),unpackFloat2x16(buf[i].a)))");
        custom_defines.append("buffer_st8(buf,i,v)", "{buf[i]=uvec4(uvec2(packFloat2x16(v[0].rg),packFloat2x16(v[0].ba)),uvec2(packFloat2x16(v[1].rg),packFloat2x16(v[1].ba)));}");
        custom_defines.append("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; f16vec2 _v0=unpackFloat2x16(_v.r);f16vec2 _v1=unpackFloat2x16(_v.g);f16vec2 _v2=unpackFloat2x16(_v.b);f16vec2 _v3=unpackFloat2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}");

        custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);buffer_st1(buf,ii4.r,_v2.r);buffer_st1(buf,ii4.g,_v2.g);buffer_st1(buf,ii4.b,_v3.r);buffer_st1(buf,ii4.a,_v3.g);}");

        custom_defines.append("buffer_cp8to4(buf,i2,sbuf,si)", "{uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}");
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.append("buffer_ld1(buf,i)", "float(buf[i])");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=float16_t(v);}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}");
        custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i].abcd.r=sbuf[si4.r];buf[i].abcd.g=sbuf[si4.g];buf[i].abcd.b=sbuf[si4.b];buf[i].abcd.a=sbuf[si4.a];buf[i].efgh.r=sbuf[sii4.r];buf[i].efgh.g=sbuf[sii4.g];buf[i].efgh.b=sbuf[sii4.b];buf[i].efgh.a=sbuf[sii4.a];}");
        custom_defines.append("buffer_ld2(buf,i)", "vec2(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=f16vec2(v);}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(buf[i])");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=f16vec4(v);}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}");
        custom_defines.append("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i].abcd=sbuf[si2.r];buf[i].efgh=sbuf[si2.g];}");
        custom_defines.append("buffer_ld8(buf,i)", "mat2x4(vec4(buf[i].abcd),vec4(buf[i].efgh))");
        custom_defines.append("buffer_st8(buf,i,v)", "{buf[i].abcd=f16vec4(v[0]);buf[i].efgh=f16vec4(v[1]);}");
        custom_defines.append("buffer_cp8(buf,i,sbuf,si)", "{buf[i].abcd=sbuf[si].abcd;buf[i].efgh=sbuf[si].efgh;}");
        custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{buf[i4.r]=sbuf[si].abcd.r;buf[i4.g]=sbuf[si].abcd.g;buf[i4.b]=sbuf[si].abcd.b;buf[i4.a]=sbuf[si].abcd.a; buf[ii4.r]=sbuf[si].efgh.r;buf[ii4.g]=sbuf[si].efgh.g;buf[ii4.b]=sbuf[si].efgh.b;buf[ii4.a]=sbuf[si].efgh.a;}");
        custom_defines.append("buffer_cp8to4(buf,i2,sbuf,si)", "{buf[i2.r]=sbuf[si].abcd;buf[i2.g]=sbuf[si].efgh;}");
    }
    else if (opt.use_fp16_packed)
    {
        // custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_ld1(buf,i)", "unpackHalf2x16(buf[(i)/2])[(i)%2]");
        // custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        // custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        // custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])));}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        // custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=uvec4(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])),packHalf2x16(vec2(sbuf[sii4.r],sbuf[sii4.g])),packHalf2x16(vec2(sbuf[sii4.b],sbuf[sii4.a])));}");

        custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _sii4d2=uvec4(sii4)/2;uvec4 _si4m2=uvec4(si4)%2;uvec4 _sii4m2=uvec4(sii4)%2; buf[i]=uvec4(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_sii4d2.r])[_sii4m2.r],unpackHalf2x16(sbuf[_sii4d2.g])[_sii4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_sii4d2.b])[_sii4m2.b],unpackHalf2x16(sbuf[_sii4d2.a])[_sii4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackHalf2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packHalf2x16(v)}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");

        custom_defines.append("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}");
        custom_defines.append("buffer_ld8(buf,i)", "mat2x4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g)),vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a)))");
        custom_defines.append("buffer_st8(buf,i,v)", "{buf[i]=uvec4(uvec2(packHalf2x16(v[0].rg),packHalf2x16(v[0].ba)),uvec2(packHalf2x16(v[1].rg),packHalf2x16(v[1].ba)));}");
        custom_defines.append("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}");

        custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);buffer_st1(buf,ii4.r,_v2.r);buffer_st1(buf,ii4.g,_v2.g);buffer_st1(buf,ii4.b,_v3.r);buffer_st1(buf,ii4.a,_v3.g);}");

        custom_defines.append("buffer_cp8to4(buf,i2,sbuf,si)", "{uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}");
    }
    else
    {
        custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}");
        custom_defines.append("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}");
        custom_defines.append("buffer_ld2(buf,i)", "buf[i]");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "buf[i]");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{vec4 _v=sbuf[si]; buf[i4.r]=_v.r;buf[i4.g]=_v.g;buf[i4.b]=_v.b;buf[i4.a]=_v.a;}");
        custom_defines.append("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=mat2x4(sbuf[si2.r],sbuf[si2.g]);}");
        custom_defines.append("buffer_ld8(buf,i)", "buf[i]");
        custom_defines.append("buffer_st8(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}");
        custom_defines.append("buffer_cp8to4(buf,i2,sbuf,si)", "{mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}");
        custom_defines.append("sfp2afpmat4(v)", "v");
        custom_defines.append("afp2sfpmat4(v)", "v");
    }

    if (opt.use_int8_storage)
    {
        custom_defines.append("sint8", "int8_t");
    }
    else if (opt.use_int8_packed)
    {
        custom_defines.append("sint8", "int");
    }
    else
    {
        custom_defines.append("sint8", "int");
    }

    custom_defines.append("sint8vec4", "int");
    custom_defines.append("sint8vec8", "ivec2");

    custom_defines.append("aint8", "int");
    custom_defines.append("aint8vec4", "ivec4");

    custom_defines.append("unpackInt4x8(v)", "ivec4((v<<24)>>24,(v<<16)>>24,(v<<8)>>24,v>>24)");
    custom_defines.append("packInt4x8(v)", "int((uint(v.r)&0xFFu)|((uint(v.g)&0xFFu)<<8)|((uint(v.b)&0xFFu)<<16)|((uint(v.a)&0xFFu)<<24))");

    if (opt.use_int8_storage)
    {
        custom_defines.append("i8buffer_ld1(buf,i)", "int(buf[i])");
        custom_defines.append("i8buffer_st1(buf,i,v)", "{buf[i]=int8_t(v);}");
        custom_defines.append("i8buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
    }
    else
    {
        custom_defines.append("i8buffer_ld1(buf,i)", "int(((buf[(i)/4])<<(24-((i)%4)*8))>>24)");
        custom_defines.append("i8buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id4=_i/4;uint _im4=_i%4;int _vs=int(v);int _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id4],0,0);ivec4 _v=unpackInt4x8(_old_v);_v[_im4]=_vs;_new_v=packInt4x8(_v);} while(atomicCompSwap(buf[_id4],_old_v,_new_v)!=_old_v);}");
        custom_defines.append("i8buffer_cp1(buf,i,sbuf,si)", "{int _v=i8buffer_ld1(sbuf,si);i8buffer_st1(buf,i,_v);}");
    }

    custom_defines.append("i8buffer_ld4(buf,i)", "unpackInt4x8(buf[i])");
    custom_defines.append("i8buffer_st4(buf,i,v)", "{buf[i]=packInt4x8(v);}");
    custom_defines.append("i8buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

    custom_defines.append("i8buffer_ld8(buf,i)", "ivec8(unpackInt4x8(buf[i].r),unpackInt4x8(buf[i].g))");
    custom_defines.append("i8buffer_st8(buf,i,v)", "{buf[i]=ivec2(packInt4x8(v.abcd),packInt4x8(v.efgh));}");
    custom_defines.append("i8buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

    custom_defines.append("psc(x)", "(x==0?p.x:x)");

    if (opt.use_fp16_storage)
    {
        custom_defines.append("NCNN_fp16_storage", 1);
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("NCNN_fp16_packed", 1);
    }

    if (opt.use_fp16_uniform)
    {
        custom_defines.append("NCNN_fp16_uniform", 1);
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.append("NCNN_fp16_arithmetic", 1);
    }

    if (opt.use_int8_storage)
    {
        custom_defines.append("NCNN_int8_storage", 1);
    }
    else if (opt.use_int8_packed)
    {
        custom_defines.append("NCNN_int8_packed", 1);
    }

    if (opt.use_int8_uniform)
    {
        custom_defines.append("NCNN_int8_uniform", 1);
    }

    if (opt.use_int8_arithmetic)
    {
        custom_defines.append("NCNN_int8_arithmetic", 1);
    }

    if (opt.use_shader_local_memory)
    {
        custom_defines.append("NCNN_shader_local_memory", 1);
    }

#if __APPLE__
    custom_defines.append("NCNN_moltenvk", 1);
#endif

    custom_defines.append("ncnn_glsl_version", 1);

    bool support_shader_int64 = false;

    // fill device macros
    {
        int device_index = opt.vulkan_device_index;
        if (device_index < 0 || device_index >= get_gpu_count())
            device_index = get_default_gpu_index();

        const GpuInfo& info = get_gpu_info(device_index);

        support_shader_int64 = info.physicalDevicefeatures().shaderInt64;

        // pull in device extensions
        {
            const std::vector<VkExtensionProperties>& properties = info.deviceExtensionProperties();

            for (size_t i = 0; i < properties.size(); i++)
            {
                const VkExtensionProperties& exp = properties[i];
                device_defines.append(exp.extensionName, exp.specVersion);
            }
        }

#define DD_APPEND_FEATURE(X) device_defines.append(#X, features.X ? 1 : 0);

        // pull in device features macros
        {
            const VkPhysicalDeviceFeatures& features = info.physicalDevicefeatures();
            DD_APPEND_FEATURE(robustBufferAccess)
            DD_APPEND_FEATURE(fullDrawIndexUint32)
            DD_APPEND_FEATURE(imageCubeArray)
            DD_APPEND_FEATURE(independentBlend)
            DD_APPEND_FEATURE(geometryShader)
            DD_APPEND_FEATURE(tessellationShader)
            DD_APPEND_FEATURE(sampleRateShading)
            DD_APPEND_FEATURE(dualSrcBlend)
            DD_APPEND_FEATURE(logicOp)
            DD_APPEND_FEATURE(multiDrawIndirect)
            DD_APPEND_FEATURE(drawIndirectFirstInstance)
            DD_APPEND_FEATURE(depthClamp)
            DD_APPEND_FEATURE(depthBiasClamp)
            DD_APPEND_FEATURE(fillModeNonSolid)
            DD_APPEND_FEATURE(depthBounds)
            DD_APPEND_FEATURE(wideLines)
            DD_APPEND_FEATURE(largePoints)
            DD_APPEND_FEATURE(alphaToOne)
            DD_APPEND_FEATURE(multiViewport)
            DD_APPEND_FEATURE(samplerAnisotropy)
            DD_APPEND_FEATURE(textureCompressionETC2)
            DD_APPEND_FEATURE(textureCompressionASTC_LDR)
            DD_APPEND_FEATURE(textureCompressionBC)
            DD_APPEND_FEATURE(occlusionQueryPrecise)
            DD_APPEND_FEATURE(pipelineStatisticsQuery)
            DD_APPEND_FEATURE(vertexPipelineStoresAndAtomics)
            DD_APPEND_FEATURE(fragmentStoresAndAtomics)
            DD_APPEND_FEATURE(shaderTessellationAndGeometryPointSize)
            DD_APPEND_FEATURE(shaderImageGatherExtended)
            DD_APPEND_FEATURE(shaderStorageImageExtendedFormats)
            DD_APPEND_FEATURE(shaderStorageImageMultisample)
            DD_APPEND_FEATURE(shaderStorageImageReadWithoutFormat)
            DD_APPEND_FEATURE(shaderStorageImageWriteWithoutFormat)
            DD_APPEND_FEATURE(shaderUniformBufferArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderSampledImageArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderStorageBufferArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderStorageImageArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderClipDistance)
            DD_APPEND_FEATURE(shaderCullDistance)
            DD_APPEND_FEATURE(shaderFloat64)
            DD_APPEND_FEATURE(shaderInt64)
            DD_APPEND_FEATURE(shaderInt16)
            DD_APPEND_FEATURE(shaderResourceResidency)
            DD_APPEND_FEATURE(shaderResourceMinLod)
            DD_APPEND_FEATURE(sparseBinding)
            DD_APPEND_FEATURE(sparseResidencyBuffer)
            DD_APPEND_FEATURE(sparseResidencyImage2D)
            DD_APPEND_FEATURE(sparseResidencyImage3D)
            DD_APPEND_FEATURE(sparseResidency2Samples)
            DD_APPEND_FEATURE(sparseResidency4Samples)
            DD_APPEND_FEATURE(sparseResidency8Samples)
            DD_APPEND_FEATURE(sparseResidency16Samples)
            DD_APPEND_FEATURE(sparseResidencyAliased)
            DD_APPEND_FEATURE(variableMultisampleRate)
            DD_APPEND_FEATURE(inheritedQueries)
        }
        if (info.support_VK_KHR_8bit_storage())
        {
            const VkPhysicalDevice8BitStorageFeaturesKHR& features = info.query8BitStorageFeatures();
            DD_APPEND_FEATURE(storageBuffer8BitAccess)
            DD_APPEND_FEATURE(uniformAndStorageBuffer8BitAccess)
            DD_APPEND_FEATURE(storagePushConstant8)
        }
        if (info.support_VK_KHR_16bit_storage())
        {
            const VkPhysicalDevice16BitStorageFeaturesKHR& features = info.query16BitStorageFeatures();
            DD_APPEND_FEATURE(storageBuffer16BitAccess)
            DD_APPEND_FEATURE(uniformAndStorageBuffer16BitAccess)
            DD_APPEND_FEATURE(storagePushConstant16)
            DD_APPEND_FEATURE(storageInputOutput16)
        }
        if (info.support_VK_KHR_shader_float16_int8())
        {
            const VkPhysicalDeviceFloat16Int8FeaturesKHR& features = info.queryFloat16Int8Features();
            DD_APPEND_FEATURE(shaderFloat16)
            DD_APPEND_FEATURE(shaderInt8)
        }
        if (info.support_VK_KHR_sampler_ycbcr_conversion())
        {
            const VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR& features = info.querySamplerYcbcrConversionFeatures();
            DD_APPEND_FEATURE(samplerYcbcrConversion)
        }
        if (info.support_VK_KHR_cooperative_matrix())
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesKHR& features = info.queryCooperativeMatrixFeatures();
            DD_APPEND_FEATURE(cooperativeMatrix)
            DD_APPEND_FEATURE(cooperativeMatrixRobustBufferAccess)
        }
        else if (info.support_VK_NV_cooperative_matrix())
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesNV& features = info.queryCooperativeMatrixFeaturesNV();
            DD_APPEND_FEATURE(cooperativeMatrix)
            DD_APPEND_FEATURE(cooperativeMatrixRobustBufferAccess)
        }
        if (info.support_VK_NV_cooperative_matrix2())
        {
            const VkPhysicalDeviceCooperativeMatrix2FeaturesNV& features = info.queryCooperativeMatrix2FeaturesNV();
            DD_APPEND_FEATURE(cooperativeMatrixWorkgroupScope)
            DD_APPEND_FEATURE(cooperativeMatrixFlexibleDimensions)
            DD_APPEND_FEATURE(cooperativeMatrixReductions)
            DD_APPEND_FEATURE(cooperativeMatrixConversions)
            DD_APPEND_FEATURE(cooperativeMatrixPerElementOperations)
            DD_APPEND_FEATURE(cooperativeMatrixTensorAddressing)
            DD_APPEND_FEATURE(cooperativeMatrixBlockLoads)
        }
        if (info.support_VK_NV_cooperative_vector())
        {
            const VkPhysicalDeviceCooperativeVectorFeaturesNV& features = info.queryCooperativeVectorFeaturesNV();
            DD_APPEND_FEATURE(cooperativeVector)
            DD_APPEND_FEATURE(cooperativeVectorTraining)
        }
        if (info.support_VK_EXT_subgroup_size_control())
        {
            const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT& features = info.querySubgroupSizeControlFeatures();
            DD_APPEND_FEATURE(subgroupSizeControl)
            DD_APPEND_FEATURE(computeFullSubgroups)
        }
        if (info.support_VK_KHR_shader_bfloat16())
        {
            const VkPhysicalDeviceShaderBfloat16FeaturesKHR& features = info.queryShaderBfloat16Features();
            DD_APPEND_FEATURE(shaderBFloat16Type)
            DD_APPEND_FEATURE(shaderBFloat16DotProduct)
            DD_APPEND_FEATURE(shaderBFloat16CooperativeMatrix)
        }
        if (info.support_VK_EXT_shader_float8())
        {
            const VkPhysicalDeviceShaderFloat8FeaturesEXT& features = info.queryShaderFloat8Features();
            DD_APPEND_FEATURE(shaderFloat8)
            DD_APPEND_FEATURE(shaderFloat8CooperativeMatrix)
        }
        if (info.support_VK_KHR_shader_float_controls2())
        {
            const VkPhysicalDeviceShaderFloatControls2FeaturesKHR& features = info.queryShaderFloatControls2Features();
            DD_APPEND_FEATURE(shaderFloatControls2)
        }
        if (info.support_VK_KHR_shader_integer_dot_product())
        {
            const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR& features = info.queryShaderIntegerDotProductFeatures();
            DD_APPEND_FEATURE(shaderIntegerDotProduct)
        }
        if (info.support_VK_KHR_shader_subgroup_rotate())
        {
            const VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR& features = info.queryShaderSubgroupRotateFeatures();
            DD_APPEND_FEATURE(shaderSubgroupRotate)
            DD_APPEND_FEATURE(shaderSubgroupRotateClustered)
        }
        if (info.support_VK_EXT_shader_atomic_float())
        {
            const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& features = info.queryShaderAtomicFloatFeatures();
            DD_APPEND_FEATURE(shaderBufferFloat32Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat32AtomicAdd)
            DD_APPEND_FEATURE(shaderBufferFloat64Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat64AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat32Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat32AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat64Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat64AtomicAdd)
            DD_APPEND_FEATURE(shaderImageFloat32Atomics)
            DD_APPEND_FEATURE(shaderImageFloat32AtomicAdd)
            DD_APPEND_FEATURE(sparseImageFloat32Atomics)
            DD_APPEND_FEATURE(sparseImageFloat32AtomicAdd)
        }
        if (info.support_VK_EXT_shader_atomic_float2())
        {
            const VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT& features = info.queryShaderAtomicFloat2Features();
            DD_APPEND_FEATURE(shaderBufferFloat16Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat16AtomicAdd)
            DD_APPEND_FEATURE(shaderBufferFloat16AtomicMinMax)
            DD_APPEND_FEATURE(shaderBufferFloat32AtomicMinMax)
            DD_APPEND_FEATURE(shaderBufferFloat64AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat16Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat16AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat16AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat32AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat64AtomicMinMax)
            DD_APPEND_FEATURE(shaderImageFloat32AtomicMinMax)
            DD_APPEND_FEATURE(sparseImageFloat32AtomicMinMax)
        }
        if (info.support_VK_KHR_vulkan_memory_model())
        {
            const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR& features = info.queryVulkanMemoryModelFeatures();
            DD_APPEND_FEATURE(vulkanMemoryModel)
            DD_APPEND_FEATURE(vulkanMemoryModelDeviceScope)
            DD_APPEND_FEATURE(vulkanMemoryModelAvailabilityVisibilityChains)
        }

#undef DD_APPEND_FEATURE

#define DD_APPEND_PROPERTY(X) device_defines.append(#X, properties.X);

        // pull in device properties macros
        {
            const VkPhysicalDeviceProperties& properties = info.physicalDeviceProperties();
            DD_APPEND_PROPERTY(apiVersion)
            DD_APPEND_PROPERTY(driverVersion)
            DD_APPEND_PROPERTY(vendorID)
            DD_APPEND_PROPERTY(deviceID)
            DD_APPEND_PROPERTY(deviceType)
            // DD_APPEND_PROPERTY(deviceName)

            // DD_APPEND_PROPERTY(pipelineCacheUUID)

#define DD_APPEND_PROPERTY_LIMIT(X) device_defines.append(#X, properties.limits.X);
#define DD_APPEND_PROPERTY_LIMIT_2(X)                       \
    device_defines.append(#X "_0", properties.limits.X[0]); \
    device_defines.append(#X "_1", properties.limits.X[1]);
#define DD_APPEND_PROPERTY_LIMIT_3(X)                       \
    device_defines.append(#X "_0", properties.limits.X[0]); \
    device_defines.append(#X "_1", properties.limits.X[1]); \
    device_defines.append(#X "_2", properties.limits.X[2]);

            DD_APPEND_PROPERTY_LIMIT(maxImageDimension1D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimension2D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimension3D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimensionCube)
            DD_APPEND_PROPERTY_LIMIT(maxImageArrayLayers)
            DD_APPEND_PROPERTY_LIMIT(maxTexelBufferElements)
            DD_APPEND_PROPERTY_LIMIT(maxUniformBufferRange)
            DD_APPEND_PROPERTY_LIMIT(maxStorageBufferRange)
            DD_APPEND_PROPERTY_LIMIT(maxPushConstantsSize)
            DD_APPEND_PROPERTY_LIMIT(maxMemoryAllocationCount)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerAllocationCount)
            DD_APPEND_PROPERTY_LIMIT(bufferImageGranularity)
            DD_APPEND_PROPERTY_LIMIT(sparseAddressSpaceSize)
            DD_APPEND_PROPERTY_LIMIT(maxBoundDescriptorSets)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorSamplers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorUniformBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorStorageBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorSampledImages)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorStorageImages)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorInputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageResources)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetSamplers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetUniformBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetUniformBuffersDynamic)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageBuffersDynamic)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetSampledImages)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageImages)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetInputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputAttributes)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputBindings)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputAttributeOffset)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputBindingStride)
            DD_APPEND_PROPERTY_LIMIT(maxVertexOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationGenerationLevel)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationPatchSize)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerVertexInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerVertexOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerPatchOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlTotalOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationEvaluationInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationEvaluationOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryShaderInvocations)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryOutputVertices)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryTotalOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentOutputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentDualSrcAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentCombinedOutputResources)
            DD_APPEND_PROPERTY_LIMIT(maxComputeSharedMemorySize)
            DD_APPEND_PROPERTY_LIMIT_3(maxComputeWorkGroupCount)
            DD_APPEND_PROPERTY_LIMIT(maxComputeWorkGroupInvocations)
            DD_APPEND_PROPERTY_LIMIT_3(maxComputeWorkGroupSize)
            DD_APPEND_PROPERTY_LIMIT(subPixelPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(subTexelPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(mipmapPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(maxDrawIndexedIndexValue)
            DD_APPEND_PROPERTY_LIMIT(maxDrawIndirectCount)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerLodBias)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerAnisotropy)
            DD_APPEND_PROPERTY_LIMIT(maxViewports)
            DD_APPEND_PROPERTY_LIMIT_2(maxViewportDimensions)
            DD_APPEND_PROPERTY_LIMIT_2(viewportBoundsRange)
            DD_APPEND_PROPERTY_LIMIT(viewportSubPixelBits)
            device_defines.append("minMemoryMapAlignment", (uint32_t)properties.limits.minMemoryMapAlignment);
            DD_APPEND_PROPERTY_LIMIT(minTexelBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minUniformBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minStorageBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minTexelOffset)
            DD_APPEND_PROPERTY_LIMIT(maxTexelOffset)
            DD_APPEND_PROPERTY_LIMIT(minTexelGatherOffset)
            DD_APPEND_PROPERTY_LIMIT(maxTexelGatherOffset)
            DD_APPEND_PROPERTY_LIMIT(minInterpolationOffset)
            DD_APPEND_PROPERTY_LIMIT(maxInterpolationOffset)
            DD_APPEND_PROPERTY_LIMIT(subPixelInterpolationOffsetBits)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferWidth)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferHeight)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferLayers)
            DD_APPEND_PROPERTY_LIMIT(framebufferColorSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferDepthSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferStencilSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferNoAttachmentsSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(maxColorAttachments)
            DD_APPEND_PROPERTY_LIMIT(sampledImageColorSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageIntegerSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageDepthSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageStencilSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(storageImageSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(maxSampleMaskWords)
            DD_APPEND_PROPERTY_LIMIT(timestampComputeAndGraphics)
            DD_APPEND_PROPERTY_LIMIT(timestampPeriod)
            DD_APPEND_PROPERTY_LIMIT(maxClipDistances)
            DD_APPEND_PROPERTY_LIMIT(maxCullDistances)
            DD_APPEND_PROPERTY_LIMIT(maxCombinedClipAndCullDistances)
            DD_APPEND_PROPERTY_LIMIT(discreteQueuePriorities)
            DD_APPEND_PROPERTY_LIMIT_2(pointSizeRange)
            DD_APPEND_PROPERTY_LIMIT_2(lineWidthRange)
            DD_APPEND_PROPERTY_LIMIT(pointSizeGranularity)
            DD_APPEND_PROPERTY_LIMIT(lineWidthGranularity)
            DD_APPEND_PROPERTY_LIMIT(strictLines)
            DD_APPEND_PROPERTY_LIMIT(standardSampleLocations)
            DD_APPEND_PROPERTY_LIMIT(optimalBufferCopyOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(optimalBufferCopyRowPitchAlignment)
            DD_APPEND_PROPERTY_LIMIT(nonCoherentAtomSize)

#undef DD_APPEND_PROPERTY_LIMIT
#undef DD_APPEND_PROPERTY_LIMIT_2
#undef DD_APPEND_PROPERTY_LIMIT_3

#define DD_APPEND_PROPERTY_SPARSE(X) device_defines.append(#X, properties.sparseProperties.X);

            DD_APPEND_PROPERTY_SPARSE(residencyStandard2DBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyStandard2DMultisampleBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyStandard3DBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyAlignedMipSize)
            DD_APPEND_PROPERTY_SPARSE(residencyNonResidentStrict)

#undef DD_APPEND_PROPERTY_SPARSE
        }
        {
            const VkPhysicalDeviceSubgroupProperties& properties = info.querySubgroupProperties();
            DD_APPEND_PROPERTY(subgroupSize)
            DD_APPEND_PROPERTY(supportedStages)
            DD_APPEND_PROPERTY(supportedOperations)
            DD_APPEND_PROPERTY(quadOperationsInAllStages)

            // append subgroup ops
            device_defines.append("subgroup_basic", (properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ? 1 : 0);
            device_defines.append("subgroup_vote", (properties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) ? 1 : 0);
            device_defines.append("subgroup_arithmetic", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? 1 : 0);
            device_defines.append("subgroup_ballot", (properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? 1 : 0);
            device_defines.append("subgroup_shuffle", (properties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? 1 : 0);
            device_defines.append("subgroup_shuffle_relative", (properties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) ? 1 : 0);
            device_defines.append("subgroup_clustered", (properties.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) ? 1 : 0);
            device_defines.append("subgroup_quad", (properties.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) ? 1 : 0);
            device_defines.append("subgroup_rotate", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ROTATE_BIT) ? 1 : 0);
            device_defines.append("subgroup_rotate_relative", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT) ? 1 : 0);
            device_defines.append("subgroup_partitioned", (properties.supportedOperations & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV) ? 1 : 0);
        }
        if (info.support_VK_NV_cooperative_matrix2())
        {
            const VkPhysicalDeviceCooperativeMatrix2PropertiesNV& properties = info.queryCooperativeMatrix2PropertiesNV();
            DD_APPEND_PROPERTY(cooperativeMatrixWorkgroupScopeMaxWorkgroupSize)
            DD_APPEND_PROPERTY(cooperativeMatrixFlexibleDimensionsMaxDimension)
            DD_APPEND_PROPERTY(cooperativeMatrixWorkgroupScopeReservedSharedMemory)
        }
        if (info.support_VK_NV_cooperative_vector())
        {
            const VkPhysicalDeviceCooperativeVectorPropertiesNV& properties = info.queryCooperativeVectorPropertiesNV();
            DD_APPEND_PROPERTY(cooperativeVectorSupportedStages)
            DD_APPEND_PROPERTY(cooperativeVectorTrainingFloat16Accumulation)
            DD_APPEND_PROPERTY(cooperativeVectorTrainingFloat32Accumulation)
            DD_APPEND_PROPERTY(maxCooperativeVectorComponents)
        }
        if (info.support_VK_KHR_driver_properties())
        {
            const VkPhysicalDeviceDriverPropertiesKHR& properties = info.queryDriverProperties();
            DD_APPEND_PROPERTY(driverID)
            // DD_APPEND_PROPERTY(driverName)
            // DD_APPEND_PROPERTY(driverInfo)
            device_defines.append("conformanceVersion_major", properties.conformanceVersion.major);
            device_defines.append("conformanceVersion_minor", properties.conformanceVersion.minor);
            device_defines.append("conformanceVersion_subminor", properties.conformanceVersion.subminor);
            device_defines.append("conformanceVersion_patch", properties.conformanceVersion.patch);
        }
        if (info.support_VK_KHR_shader_integer_dot_product())
        {
            const VkPhysicalDeviceShaderIntegerDotProductProperties& properties = info.queryShaderIntegerDotProductProperties();
            DD_APPEND_PROPERTY(integerDotProduct8BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct8BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct8BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated)
        }
        if (info.support_VK_EXT_subgroup_size_control())
        {
            const VkPhysicalDeviceSubgroupSizeControlPropertiesEXT& properties = info.querySubgroupSizeControlProperties();
            DD_APPEND_PROPERTY(minSubgroupSize)
            DD_APPEND_PROPERTY(maxSubgroupSize)
            DD_APPEND_PROPERTY(maxComputeWorkgroupSubgroups)
            DD_APPEND_PROPERTY(requiredSubgroupSizeStages)
        }

#if ENABLE_VALIDATION_LAYER
        if (info.support_VK_KHR_shader_non_semantic_info())
        {
            device_defines.append("enable_validation_layer", VK_TRUE);
            custom_defines.append("NCNN_LOGE", "debugPrintfEXT");
        }
#endif

#undef DD_APPEND_PROPERTY
    }

    std::string define_macro_data;

    for (size_t i = 0; i < custom_defines.definitions.size(); i++)
    {
        const char* key = custom_defines.definitions[i].first;
        const DefinitionCollector::typed_value& def = custom_defines.definitions[i].second;

        if (def.type == 0)
        {
            define_macro_data += std::string("#define ") + key + " " + def.s + "\n";
        }
        else
        {
            char defstr[256];
            if (def.type == 1)
            {
                sprintf(defstr, "%u", def.u8);
            }
            if (def.type == 2)
            {
                sprintf(defstr, "%u", def.u32);
            }
            if (def.type == 3)
            {
                sprintf(defstr, "%d", def.i32);
            }
            if (def.type == 4)
            {
                if (support_shader_int64)
                {
                    sprintf(defstr, "%luull", def.u64);
                }
                else
                {
                    uint32_t u32 = def.u64 > UINT_MAX ? UINT_MAX : (uint32_t)def.u64;
                    sprintf(defstr, "%u", u32);
                }
            }
            if (def.type == 5)
            {
                sprintf(defstr, "%e", def.f32);
            }

            define_macro_data += std::string("#define ") + key + " " + defstr + "\n";
        }
    }
    for (size_t i = 0; i < device_defines.definitions.size(); i++)
    {
        const char* key = device_defines.definitions[i].first;
        const DefinitionCollector::typed_value& def = device_defines.definitions[i].second;

        if (def.type == 0)
        {
            define_macro_data += std::string("#define ncnn_") + key + " \"" + def.s + "\"\n";
        }
        else
        {
            char defstr[256];
            if (def.type == 1)
            {
                sprintf(defstr, "%u", def.u8);
            }
            if (def.type == 2)
            {
                sprintf(defstr, "%u", def.u32);
            }
            if (def.type == 3)
            {
                sprintf(defstr, "%d", def.i32);
            }
            if (def.type == 4)
            {
                if (support_shader_int64)
                {
                    sprintf(defstr, "%luull", def.u64);
                }
                else
                {
                    uint32_t u32 = def.u64 > UINT_MAX ? UINT_MAX : (uint32_t)def.u64;
                    sprintf(defstr, "%u", u32);
                }
            }
            if (def.type == 5)
            {
                sprintf(defstr, "%e", def.f32);
            }

            define_macro_data += std::string("#define ncnn_") + key + " " + defstr + "\n";
        }
    }

    // enable extensions
    std::string custom_exts;
    if (support_shader_int64)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_int64: require\n";
    }
    if (opt.use_fp16_storage)
    {
        custom_exts += "#extension GL_EXT_shader_16bit_storage: require\n";
        custom_exts += "struct sfpvec8 { f16vec4 abcd; f16vec4 efgh; };\n";
    }
    if (opt.use_fp16_arithmetic)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_float16: require\n";
    }
    custom_exts += "struct ivec8 { ivec4 abcd; ivec4 efgh; };\n";
    if (opt.use_int8_storage)
    {
        custom_exts += "#extension GL_EXT_shader_8bit_storage: require\n";
    }
    if (opt.use_int8_arithmetic)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_int8: require\n";
    }
#if ENABLE_VALIDATION_LAYER
    {
        custom_exts += "#extension GL_EXT_debug_printf : require\n";
    }
#endif

    // debug
    // NCNN_LOGE("%s", define_macro_data.c_str());

    bool compile_success = true;

    {
        glslang::TShader s(EShLangCompute);

        // split shader source by token "#version 450\n"
        int version_end_pos = -1;
        {
            for (int i = 0; i < comp_data_size - 8; i++)
            {
                if (strncmp(comp_data + i, "#version", 8) != 0)
                    continue;

                // #version shall be the very beginning or after newline
                if (i != 0 && comp_data[i - 1] != '\n')
                    continue;

                int nversion = 0;
                sscanf(comp_data + i, "#version %*d\n%n", &nversion);
                if (nversion == 0)
                    continue;

                version_end_pos = i + nversion;
                break;
            }

            if (version_end_pos == -1)
            {
                NCNN_LOGE("shader source has no #version token");
                return -1;
            }

            // NCNN_LOGE("version_end_pos = %d", version_end_pos);
        }

        const char* comp_data_2 = comp_data + version_end_pos;
        int comp_data_size_1 = version_end_pos;
        int comp_data_size_2 = comp_data_size - comp_data_size_1;

        const char* comp_datas[4] = {comp_data, custom_exts.c_str(), define_macro_data.c_str(), comp_data_2};
        const int comp_data_sizes[4] = {comp_data_size_1, (int)custom_exts.size(), (int)define_macro_data.size(), comp_data_size_2};

        s.setStringsWithLengths(comp_datas, comp_data_sizes, 4);

        s.setEntryPoint("main");
        s.setSourceEntryPoint("main");

        s.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 1);

        if (opt.use_subgroup_ops || opt.use_cooperative_matrix)
        {
            // subgroup / cooperative_matrix need vulkan-1.1 and spirv-1.3
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_3);
        }
        else
        {
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_0);
        }

        TBuiltInResource resources = get_default_TBuiltInResource();

        VulkanShaderIncluder includer;

        bool pr = s.parse(&resources, 100, ENoProfile, false, false, EShMsgDefault, includer);
        if (!pr)
        {
            NCNN_LOGE("compile spir-v module failed");
            NCNN_LOGE("%s", s.getInfoLog());
            NCNN_LOGE("%s", s.getInfoDebugLog());

            // print as line_number: code
            {
                const char* p = comp_datas[3];
                const char* line_end;
                int line_number = 1;

                while ((line_end = strchr(p, '\n')) != NULL)
                {
                    NCNN_LOGE("%d:\t%.*s", line_number++, (int)(line_end - p), p);
                    p = line_end + 1;
                }

                if (*p != '\0')
                {
                    NCNN_LOGE("%d:\t%s", line_number, p);
                }
            }

            compile_success = false;
        }
        else
        {
            glslang::TIntermediate* ir = s.getIntermediate();
            glslang::GlslangToSpv(*ir, spirv);
        }
    }

    return compile_success ? 0 : -1;
}
```
