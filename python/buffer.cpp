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
void registerTypedBuffer(nb::module_& m, const char* name) {
    nb::class_<TypedBuffer<T>>(m, name)
        .def(nb::init<size_t>())
        .def_prop_ro("size", [](const TypedBuffer<T>& b) { return b.size(); }, "The number of elements in this buffer.")
        .def_prop_ro("size_bytes", [](const TypedBuffer<T>& b) { return b.size_bytes(); }, "The size of the buffer in bytes.")
        .def("numpy", &TypedBuffer<T>::numpy, "Returns a numpy array using this buffer's memory.");
}

void registerBufferModule(nb::module_& m) {
    nb::class_<RawBuffer>(m, "RawBuffer")
        .def(nb::init<uint64_t>())
        .def_prop_ro("address", [](const RawBuffer& b) { return b.getAddress(); } , "The memory address of the allocated buffer.")
        .def_prop_ro("size", [](const RawBuffer& b) { return b.size(); }, "The size of the buffer in bytes.");

    //Register typed arrays
    registerTypedBuffer<float>(m, "FloatBuffer");
    registerTypedBuffer<double>(m, "DoubleBuffer");
    registerTypedBuffer<uint8_t>(m, "ByteBuffer");
    registerTypedBuffer<uint16_t>(m, "UnsignedShortBuffer");
    registerTypedBuffer<uint32_t>(m, "UnsignedIntBuffer");
    registerTypedBuffer<uint64_t>(m, "UnsignedLongBuffer");
    registerTypedBuffer<int8_t>(m, "CharBuffer");
    registerTypedBuffer<int16_t>(m, "ShortBuffer");
    registerTypedBuffer<int32_t>(m, "IntBuffer");
    registerTypedBuffer<int64_t>(m, "LongBuffer");
}
