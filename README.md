# Hephaistos

Hephaistos is a framework for creating GPGPU appllication running on top of the
Vulkan API, allowing it to run on all major OS and on a wide range of hardware
such as all modern AMD, Intel and NVIDIA GPUs.

The library aims to reduce the amount of boiler plate code, albeit in doing so
it might hide optimization possibility Vulkan provides.

## Building

### C++

For building the C++ library you require:

- C++20 compatible compiler
- CMake at least version 3.18

The dependencies are located in the `external` directory.
In there is a readme file for more information.

The debug build requires the validation layers, which are part of the
[Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).

Hephaistos expects the kernel/shader code to be already compiled into SPIR-V
bytecode, so you must likely also need an compiler for that. The most popular
ones are:

- [glslangvalidator](https://github.com/KhronosGroup/glslang)
- [glslc](https://github.com/google/shaderc)

See the readme inside the `examples` folder for a hint on how to use them.

### Python

There are also python bindings using
[nanobind](https://github.com/wjakob/nanobind) and
[scikit-build](https://github.com/scikit-build/scikit-build).

To build the python package you can simply run from the project's root
directory:

```bash
pip install .
```

Note that the binding relies on the C++ library and thus has the same
dependencies. The dependencies on the Python side are handled by pip.

## Learning

There are some examples provided in the `examples` directory for C++ and
in `notebooks` for Python.

## License

Hephaistos is licensed under the MIT license. See `LICENSE`. External libraries
are distributed under their own license. The ones located in `external` are
always acompanied by a copy of their license.
