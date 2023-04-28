# C++ Examples

This directory contains examples to showcase how to use `hephaistos` with C++.

The shaders have already been compiled with [glslang](https://github.com/KhronosGroup/glslang)
using the following command:

```bash
glslangvalidator -V -g0 -o add.h --target-env vulkan1.2 --vn code add.comp
```

## List of Examples

| Name | Description |
|------|-------------|
|`add` | Adds two list of integers on the GPU. |
|`copy`| Shows how to copy to and from the GPU. |
|`deviceinfo`| Lists all suitable devices and prints their properties. |
|`mandelbrot`| Creates an image of the mandelbrot set. |
|`raytracing`| Shows how to use modern ray tracing hardware. |
|`sampling`  | Scales images via gpu interpolation. |
