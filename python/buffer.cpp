#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <sstream>
#include <string>

#include <hephaistos/buffer.hpp>
#include <hephaistos/program.hpp>

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
        .def("numpy", &TypedBuffer<T>::numpy, "Returns a numpy array using this buffer's memory.", nb::rv_policy::reference_internal)
        .def("__repr__", [name = std::string(name)](const TypedBuffer<T>& t) {
            std::ostringstream str;
            str << name
                << "[" << t.size() << "] ("
                << t.size_bytes() << " bytes)\n";
            return str.str();
        });
}

namespace {

template<class T>
std::span<const T> span_cast(const nb::ndarray<T, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>& array) {
    return { array.data(), array.size() };
}

}

template<class T>
class TypedTensor : public hp::Tensor<T> {
public:
    using array_type = nb::ndarray<T, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;

    TypedTensor(size_t count) : hp::Tensor<T>(getCurrentContext(), count) {}
    TypedTensor(uint64_t addr, size_t n) : hp::Tensor<T>(getCurrentContext(), { reinterpret_cast<const T*>(addr), n }) {}
    TypedTensor(const array_type& array) : hp::Tensor<T>(getCurrentContext(), span_cast(array)) {}
    ~TypedTensor() override = default;
};
template<class T>
void registerTensor(nb::module_& m, const char* name) {
    nb::class_<TypedTensor<T>, hp::Tensor<std::byte>>(m, name)
        .def(nb::init<size_t>())
        .def(nb::init<uint64_t, size_t>())
        .def(nb::init<const typename TypedTensor<T>::array_type&>())
        .def_prop_ro("address", [](const TypedTensor<T>& t) { return t.address(); }, "The device address of this tensor.")
        .def_prop_ro("size", [](const TypedTensor<T>& t) { return t.size(); }, "The number of elements in this tensor.")
        .def_prop_ro("size_bytes", [](const TypedTensor<T>& t) { return t.size_bytes(); }, "The size of the tensor in bytes.")
        .def("bindParameter", [](const TypedTensor<T>& t, hp::Program& p, uint32_t b) { t.bindParameter(p.getBinding(b)); } )
        .def("bindParameter", [](const TypedTensor<T>& t, hp::Program& p, std::string_view b) { t.bindParameter(p.getBinding(b)); } )
        .def("__repr__", [name = std::string(name)](const TypedTensor<T>& t) {
            std::ostringstream str;
            str << name
                << "[" << t.size() << "] ("
                << t.size_bytes() << " bytes) at 0x"
                << std::uppercase << std::hex << t.address() <<'\n';
            return str.str();
        });
}

void registerBufferModule(nb::module_& m) {
    nb::class_<hp::Buffer<std::byte>>(m, "Buffer");
    nb::class_<hp::Tensor<std::byte>>(m, "Tensor");

    nb::class_<RawBuffer, hp::Buffer<std::byte>>(m, "RawBuffer")
        .def(nb::init<uint64_t>())
        .def_prop_ro("address", [](const RawBuffer& b) { return b.getAddress(); } , "The memory address of the allocated buffer.")
        .def_prop_ro("size", [](const RawBuffer& b) { return b.size(); }, "The size of the buffer in bytes.")
        .def_prop_ro("size_bytes", [](const RawBuffer& b) { return b.size_bytes(); }, "The size of the buffer in bytes.")
        .def("__repr__", [](const RawBuffer& b) {
            std::ostringstream str;
            str << "RawBuffer: " << b.size_bytes() << " bytes\n";
            return str.str();
        });

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

    //retrieve tensor command
    nb::class_<hp::RetrieveTensorCommand, hp::Command>(m, "RetrieveTensorCommand")
        .def("__init__", [](
            hp::RetrieveTensorCommand* cmd,
            const hp::Tensor<std::byte>& src,
            const hp::Buffer<std::byte>& dst,
            std::optional<uint64_t> bufferOffset,
            std::optional<uint64_t> tensorOffset,
            std::optional<uint64_t> size) {
                hp::CopyRegion region{};
                if (bufferOffset) region.bufferOffset = *bufferOffset;
                if (tensorOffset) region.tensorOffset = *tensorOffset;
                if (size) region.size = *size;
                new (cmd) hp::RetrieveTensorCommand(src, dst, region);
            }, "src"_a, "dst"_a,
            "bufferOffset"_a.none() = nb::none(),
            "tensorOffset"_a.none() = nb::none(),
            "size"_a.none() = nb::none());
    m.def("retrieveTensor", [](
        const hp::Tensor<std::byte>& src,
        const hp::Buffer<std::byte>& dst,
        std::optional<uint64_t> bufferOffset,
        std::optional<uint64_t> tensorOffset,
        std::optional<uint64_t> size) {
            hp::CopyRegion region{};
            if (bufferOffset) region.bufferOffset = *bufferOffset;
            if (tensorOffset) region.tensorOffset = *tensorOffset;
            if (size) region.size = *size;
            return hp::retrieveTensor(src, dst, region);
        }, "src"_a, "dst"_a,
        "bufferOffset"_a.none() = nb::none(),
        "tensorOffset"_a.none() = nb::none(),
        "size"_a.none() = nb::none(),
        "Creates a command for copying the src tensor back to the destination buffer");
    //update tensor command
    nb::class_<hp::UpdateTensorCommand, hp::Command>(m, "UpdateTensorCommand")
        .def("__init__", [](
            hp::UpdateTensorCommand* cmd,
            const hp::Buffer<std::byte>& src,
            const hp::Tensor<std::byte>& dst,
            std::optional<uint64_t> bufferOffset,
            std::optional<uint64_t> tensorOffset,
            std::optional<uint64_t> size) {
                hp::CopyRegion region{};
                if (bufferOffset) region.bufferOffset = *bufferOffset;
                if (tensorOffset) region.tensorOffset = *tensorOffset;
                if (size) region.size = *size;
                new (cmd) hp::UpdateTensorCommand(src, dst, region);
            }, "src"_a, "dst"_a,
            "bufferOffset"_a.none() = nb::none(),
            "tensorOffset"_a.none() = nb::none(),
            "size"_a.none() = nb::none());
    m.def("updateTensor", [](
        const hp::Buffer<std::byte>& src,
        const hp::Tensor<std::byte>& dst,
        std::optional<uint64_t> bufferOffset,
        std::optional<uint64_t> tensorOffset,
        std::optional<uint64_t> size) {
            hp::CopyRegion region{};
            if (bufferOffset) region.bufferOffset = *bufferOffset;
            if (tensorOffset) region.tensorOffset = *tensorOffset;
            if (size) region.size = *size;
            return hp::updateTensor(src, dst, region);
        }, "src"_a, "dst"_a,
        "bufferOffset"_a.none() = nb::none(),
        "tensorOffset"_a.none() = nb::none(),
        "size"_a.none() = nb::none(),
        "Creates a command for copying the src buffer into the destination tensor");
    //clear tensor command
    nb::class_<hp::ClearTensorCommand, hp::Command>(m, "ClearTensorCommand")
        .def("__init__", [](
            hp::ClearTensorCommand* cmd,
            const hp::Tensor<std::byte>& tensor,
            std::optional<uint64_t> offset,
            std::optional<uint64_t> size,
            std::optional<uint32_t> data) {
                hp::ClearTensorCommand::Params params{};
                if (offset) params.offset = *offset;
                if (size) params.size = *size;
                if (data) params.data = *data;
                new (cmd) hp::ClearTensorCommand(tensor, params);
            }, "tensor"_a, "offset"_a.none() = nb::none(), "size"_a.none() = nb::none(), "data"_a.none() = nb::none(),
            "Creates a command for filling a tensor with constant data over a given range. Defaults to zeroing the complete tensor");
    m.def("clearTensor", [](
        const hp::Tensor<std::byte>& tensor,
        std::optional<uint64_t> offset,
        std::optional<uint64_t> size,
        std::optional<uint32_t> data) {
            hp::ClearTensorCommand::Params params{};
            if (offset) params.offset = *offset;
            if (size) params.size = *size;
            if (data) params.data = *data;
            return hp::clearTensor(tensor, params);
        }, "tensor"_a, "offset"_a.none() = nb::none(), "size"_a.none() = nb::none(), "data"_a.none() = nb::none(),
        "Creates a command for filling a tensor with constant data over a given range. Defaults to zeroing the complete tensor");
}
