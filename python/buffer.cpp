#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <hephaistos/buffer.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

class RawBuffer : public hp::Buffer<std::byte> {
public:
    int64_t getAddress() const noexcept {
        //python is weird...
        return reinterpret_cast<int64_t>(getMemory().data());
    }

    RawBuffer(uint64_t size)
        : hp::Buffer<std::byte>(getCurrentContext(), size)
    {}
    virtual ~RawBuffer() = default;
};
template<class T>
class TypedBuffer : public hp::Buffer<T> {
public:
    using array_type = nb::ndarray<nb::numpy, T, nb::shape<1, nb::any>>;

    array_type numpy() const {
        size_t shape = hp::Buffer<T>::size();
        return array_type(
            static_cast<void*>(hp::Buffer<T>::getMemory().data()),
            1, &shape);
    }

    TypedBuffer(size_t size)
        : hp::Buffer<T>(getCurrentContext(), size)
    {}
    virtual ~TypedBuffer() = default;
};
template<class T>
void registerBuffer(nb::module_& m, const char* name) {
    nb::class_<TypedBuffer<T>, hp::Buffer<std::byte>>(m, name)
        .def(nb::init<size_t>())
        .def_prop_ro("size", [](const TypedBuffer<T>& b) { return b.size(); }, "The number of elements in this buffer.")
        .def_prop_ro("size_bytes", [](const TypedBuffer<T>& b) { return b.size_bytes(); }, "The size of the buffer in bytes.")
        .def("numpy", &TypedBuffer<T>::numpy, "Returns a numpy array using this buffer's memory.");
}

template<class T>
class PyTensor : public hp::Tensor<T> {
public:
    PyTensor(size_t count) : hp::Tensor<T>(getCurrentContext(), count) {}
    virtual ~PyTensor() = default;
};
template<class T>
void registerTensor(nb::module_& m, const char* name) {
    nb::class_<PyTensor<T>, hp::Tensor<std::byte>>(m, name)
        .def(nb::init<size_t>())
        .def_prop_ro("size", [](const PyTensor<T>& b) { return b.size(); }, "The number of elements in this tensor.")
        .def_prop_ro("size_bytes", [](const PyTensor<T>& b) { return b.size_bytes(); }, "The size of the tensor in bytes.");
}

void registerBufferModule(nb::module_& m) {
    nb::class_<hp::Buffer<std::byte>>(m, "_Buffer");
    nb::class_<hp::Tensor<std::byte>>(m, "_Tensor");

    nb::class_<RawBuffer, hp::Buffer<std::byte>>(m, "RawBuffer")
        .def(nb::init<uint64_t>())
        .def_prop_ro("address", [](const RawBuffer& b) { return b.getAddress(); } , "The memory address of the allocated buffer.")
        .def_prop_ro("size", [](const RawBuffer& b) { return b.size(); }, "The size of the buffer in bytes.")
        .def_prop_ro("size_bytes", [](const RawBuffer& b) { return b.size_bytes(); }, "The size of the buffer in bytes.");

    //Register typed buffers
    registerBuffer<float>(m, "FloatBuffer");
    registerBuffer<double>(m, "DoubleBuffer");
    registerBuffer<uint8_t>(m, "ByteBuffer");
    registerBuffer<uint16_t>(m, "UnsignedShortBuffer");
    registerBuffer<uint32_t>(m, "UnsignedIntBuffer");
    registerBuffer<uint64_t>(m, "UnsignedLongBuffer");
    registerBuffer<int8_t>(m, "CharBuffer");
    registerBuffer<int16_t>(m, "ShortBuffer");
    registerBuffer<int32_t>(m, "IntBuffer");
    registerBuffer<int64_t>(m, "LongBuffer");

    //Register typed buffers
    registerTensor<float>(m, "FloatTensor");
    registerTensor<double>(m, "DoubleTensor");
    registerTensor<uint8_t>(m, "ByteTensor");
    registerTensor<uint16_t>(m, "UnsignedShortTensor");
    registerTensor<uint32_t>(m, "UnsignedIntTensor");
    registerTensor<uint64_t>(m, "UnsignedLongTensor");
    registerTensor<int8_t>(m, "CharTensor");
    registerTensor<int16_t>(m, "ShortTensor");
    registerTensor<int32_t>(m, "IntTensor");
    registerTensor<int64_t>(m, "LongTensor");

    //register copy commands
    m.def("retrieveTensor", [](hp::Tensor<std::byte>& src, hp::Buffer<std::byte>& dst)
        { return CommandHandle{ hp::retrieveTensor(src, dst) }; }, "src"_a, "dst"_a,
        "Creates a command for copying the src tensor back to the destination buffer. They must match in size.");
    m.def("updateTensor", [](hp::Buffer<std::byte>& src, hp::Tensor<std::byte>& dst)
        { return CommandHandle{ hp::updateTensor(src, dst)}; }, "src"_a, "dst"_a,
        "Creates a command for copying the src buffer into the destination tensor. They must match in size.");
}
