#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <array>
#include <cstdint>
#include <sstream>
#include <string>

#include <hephaistos/buffer.hpp>
#include <hephaistos/program.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr auto units = std::to_array({" B", " KB", " MB", " GB"});

void printSize(size_t size, std::ostream& str) {
    int dim = 0;
    for (; dim < units.size() - 1 && size > 9216; dim++, size >>= 10);
    str << size << units[dim];
}

}

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
    using array_type = nb::ndarray<T, nb::numpy, nb::shape<1, -1>>;

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
void registerBuffer(nb::module_& m, const char* name, const char* type_name) {
    //build doc string
    std::ostringstream docStream;
    docStream
        << "Buffer representing memory allocated on the host holding an array of type "
        << type_name
        << " and given size which can be accessed as numpy array."
        << "\n\nParameters\n----------\n"
        << "size: int\n"
        << "    Number of elements\n";

    nb::class_<TypedBuffer<T>, hp::Buffer<std::byte>>(m, name)
        .def(nb::init<size_t>(), "size"_a)
        .def_prop_ro("size", [](const TypedBuffer<T>& b) { return b.size(); },
            "The number of elements in this buffer.")
        .def_prop_ro("size_bytes", [](const TypedBuffer<T>& b) { return b.size_bytes(); },
            "The size of the buffer in bytes.")
        .def("numpy", &TypedBuffer<T>::numpy, nb::rv_policy::reference_internal,
            "Returns a numpy array using this buffer's memory.")
        .def("__repr__", [name = std::string(name)](const TypedBuffer<T>& t) {
            std::ostringstream str;
            str << name
                << "[" << t.size() << "] (";
            printSize(t.size_bytes(), str);
            str << ")\n";
            return str.str();
        })
        .doc() = docStream.str();
}

namespace {

template<class T>
std::span<const T> span_cast(const nb::ndarray<T, nb::shape<-1>, nb::c_contig, nb::device::cpu>& array) {
    return { array.data(), array.size() };
}

}

template<class T>
class TypedTensor : public hp::Tensor<T> {
public:
    using array_type = nb::ndarray<T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

    TypedTensor(size_t count, bool mapped)
        : hp::Tensor<T>(getCurrentContext(), count, mapped) {}
    TypedTensor(uint64_t addr, size_t n, bool mapped)
        : hp::Tensor<T>(getCurrentContext(), { reinterpret_cast<const T*>(addr), n }, mapped) {}
    TypedTensor(const array_type& array, bool mapped)
        : hp::Tensor<T>(getCurrentContext(), span_cast(array), mapped) {}
    ~TypedTensor() override = default;
};
template<class T>
void registerTensor(nb::module_& m, const char* name, const char* type_name) {
    //build doc string
    std::ostringstream docStream;
    docStream
        << "Tensor representing memory allocated on the device holding an array of type "
        << type_name
        << " and given size. If mapped can be accessed from the host using its memory "
        << "address. Mapping can optionally be requested but will be ignored if the "
        << "device does not support it. Query its support after creation via isMapped.\n";

    nb::class_<TypedTensor<T>, hp::Tensor<std::byte>>(m, name)
        .def(nb::init<size_t, bool>(),
            "size"_a, "mapped"_a = false,
            "Creates a new tensor of given size."
            "\n\nParameters\n----------\n"
            "size: int\n"
            "    Number of elements\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map memory to host address space")
        .def(nb::init<uint64_t, size_t, bool>(),
            "addr"_a, "n"_a, "mapped"_a = false,
            "Creates a new tensor and fills it with the provided data"
            "\n\nParameters\n----------\n"
            "addr: int\n"
            "    Address of the data used to fill the tensor\n"
            "n: int\n"
            "    Number of bytes to copy from addr\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map memory to host address space")
        .def(nb::init<const typename TypedTensor<T>::array_type&, bool>(),
            "array"_a, "mapped"_a = false,
            "Creates a new tensor and fills it with the provided data"
            "\n\nParameters\n----------\n"
            "array: NDArray\n"
            "    Numpy array containing the data used to fill the tensor. "
                "Its type must match the tensor's.\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map memory to host address space")
        .def_prop_ro("address", [](const TypedTensor<T>& t) { return t.address(); },
            "The device address of this tensor.")
        .def_prop_ro("isMapped", [](const TypedTensor<T>& t) { return t.isMapped(); },
            "True, if the underlying memory is writable by the CPU.")
        .def_prop_ro("memory", [](const TypedTensor<T>& t)
                { return reinterpret_cast<intptr_t>(t.getMemory().data()); },
            "Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.")
        .def_prop_ro("size", [](const TypedTensor<T>& t) { return t.size(); },
            "The number of elements in this tensor.")
        .def_prop_ro("size_bytes", [](const TypedTensor<T>& t) { return t.size_bytes(); },
            "The size of the tensor in bytes.")
        .def_prop_ro("isNonCoherent", [](const TypedTensor<T>& t) { return t.isNonCoherent(); },
            "Wether calls to flush() and invalidate() are necessary to make changes"
            "in mapped memory between devices and host available")
        .def("update",
            [](TypedTensor<T>& t, uint64_t ptr, size_t n, uint64_t offset) {
                t.update({ reinterpret_cast<const T*>(ptr), n }, offset);
            }, "addr"_a, "n"_a, "offset"_a = 0,
            "Updates the tensor at the given offset with n elements from addr"
            "\n\nParameters\n----------\n"
            "ptr: int\n"
            "   Address of memory to copy from\n"
            "n: int\n"
            "   Amount of elements to copy\n"
            "offset: int, default=0\n"
            "   Offset into the tensor in number of elements where the copy starts")
        .def("flush",
            [](TypedTensor<T>& t, uint64_t offset, std::optional<uint64_t> size) {
                uint64_t _size = hp::whole_size;
                if (size) _size = *size;
                t.flush(offset, _size);
            }, "offset"_a = 0, "size"_a.none() = nb::none(),
            "Makes writes in mapped memory from the host available to the device"
            "\n\nParameters\n----------\n"
            "offset: int, default=0\n"
            "   Offset in amount of elements into mapped memory to flush\n"
            "size: int | None, default=None\n"
            "   Number of elements to flush starting at offset. If None, flushes all remaining elements")
        .def("retrieve",
            [](TypedTensor<T>& t, uint64_t ptr, size_t n, uint64_t offset) {
                t.retrieve({ reinterpret_cast<T*>(ptr), n }, offset);
            }, "addr"_a, "n"_a, "offset"_a = 0,
            "Retrieves data from the tensor at the given offset and "
            "stores it in dst. Only needed if isNonCoherent is True."
            "\n\nParameters\n----------\n"
            "ptr: int\n"
            "   Address of memory to copy to\n"
            "n: int\n"
            "   Amount of elements to copy\n"
            "offset: int, default=0\n"
            "   Offset into the tensor in amount of elements where the copy starts")
        .def("invalidate",
            [](TypedTensor<T>& t, uint64_t offset, std::optional<uint64_t> size) {
                uint64_t _size = hp::whole_size;
                if (size) _size = *size;
                t.invalidate(offset, _size);
            }, "offset"_a = 0, "size"_a.none() = nb::none(),
            "Makes writes in mapped memory from the device available to the host. "
            "Only needed if isNonCoherent is True."
            "\n\nParameters\n----------\n"
            "offset: int, default=0\n"
            "   Offset in amount of elements into mapped memory to invalidate\n"
            "size: int | None, default=None\n"
            "   Number of elements to invalidate starting at offset. "
               "If None, invalidates all remaining bytes")
        .def("bindParameter", [](const TypedTensor<T>& t, hp::Program& p, uint32_t b)
            { t.bindParameter(p.getBinding(b)); }, "program"_a, "binding"_a,
            "Binds the tensor to the program at the given binding")
        .def("bindParameter", [](const TypedTensor<T>& t, hp::Program& p, std::string_view b)
            { t.bindParameter(p.getBinding(b)); }, "program"_a, "binding"_a,
            "Binds the tensor to the program at the given binding")
        .def("__repr__", [name = std::string(name)](const TypedTensor<T>& t) {
            std::ostringstream str;
            str << name
                << "[" << t.size() << "] (";
            printSize(t.size_bytes(), str);
            str << ") at 0x" << std::uppercase << std::hex << t.address();
            if (t.isMapped())
                str << " (mapped)\n";
            else
                str << '\n';
            return str.str();
        })
        .doc() = docStream.str();
}

void registerBufferModule(nb::module_& m) {
    nb::class_<hp::Buffer<std::byte>>(m, "Buffer",
        "Base class for all buffers managing memory allocation on the host");
    nb::class_<hp::Tensor<std::byte>>(m, "Tensor",
        "Base class for all tensors managing memory allocations on the device");

    nb::class_<RawBuffer, hp::Buffer<std::byte>>(m, "RawBuffer",
            "Buffer for allocating a raw chunk of memory on the host "
            "accessible via its memory address. "
            "Useful as a base class providing more complex functionality."
            "\n\nParameters\n----------\n"
            "size: int\n"
            "    size of the buffer in bytes")
        .def(nb::init<uint64_t>(), "size"_a)
        .def_prop_ro("address", [](const RawBuffer& b) { return b.getAddress(); },
            "The memory address of the allocated buffer.")
        .def_prop_ro("size", [](const RawBuffer& b) { return b.size(); },
            "The size of the buffer in bytes.")
        .def_prop_ro("size_bytes", [](const RawBuffer& b) { return b.size_bytes(); },
            "The size of the buffer in bytes.")
        .def("__repr__", [](const RawBuffer& b) {
            std::ostringstream str;
            str << "RawBuffer: ";
            printSize(b.size_bytes(), str);
            str << '\n';
            return str.str();
        });

    //Register typed buffers
    registerBuffer<float>(m, "FloatBuffer", "float");
    registerBuffer<double>(m, "DoubleBuffer", "double");
    registerBuffer<uint8_t>(m, "ByteBuffer", "uint8");
    registerBuffer<uint16_t>(m, "UnsignedShortBuffer", "uint16");
    registerBuffer<uint32_t>(m, "UnsignedIntBuffer", "uint32");
    registerBuffer<uint64_t>(m, "UnsignedLongBuffer", "uint64");
    registerBuffer<int8_t>(m, "CharBuffer", "int8");
    registerBuffer<int16_t>(m, "ShortBuffer", "int16");
    registerBuffer<int32_t>(m, "IntBuffer", "int32");
    registerBuffer<int64_t>(m, "LongBuffer", "int64");

    //Register typed buffers
    registerTensor<float>(m, "FloatTensor", "float");
    registerTensor<double>(m, "DoubleTensor", "double");
    registerTensor<uint8_t>(m, "ByteTensor", "uint8");
    registerTensor<uint16_t>(m, "UnsignedShortTensor", "uint16");
    registerTensor<uint32_t>(m, "UnsignedIntTensor", "uint32");
    registerTensor<uint64_t>(m, "UnsignedLongTensor", "uint64");
    registerTensor<int8_t>(m, "CharTensor", "int8");
    registerTensor<int16_t>(m, "ShortTensor", "int16");
    registerTensor<int32_t>(m, "IntTensor", "int32");
    registerTensor<int64_t>(m, "LongTensor", "int64");

    //retrieve tensor command
    nb::class_<hp::RetrieveTensorCommand, hp::Command>(m, "RetrieveTensorCommand",
            "Command for copying the src tensor back to the destination buffer")
        .def("__init__",
            [](
                hp::RetrieveTensorCommand* cmd,
                const hp::Tensor<std::byte>& src,
                const hp::Buffer<std::byte>& dst,
                std::optional<uint64_t> bufferOffset,
                std::optional<uint64_t> tensorOffset,
                std::optional<uint64_t> size,
                bool unsafe
            ) {
                hp::CopyRegion region{};
                if (bufferOffset) region.bufferOffset = *bufferOffset;
                if (tensorOffset) region.tensorOffset = *tensorOffset;
                if (size) region.size = *size;
                region.unsafe = unsafe;
                new (cmd) hp::RetrieveTensorCommand(src, dst, region);
            }, "src"_a, "dst"_a,
            "bufferOffset"_a.none() = nb::none(),
            "tensorOffset"_a.none() = nb::none(),
            "size"_a.none() = nb::none(),
            "unsafe"_a = false);
    m.def("retrieveTensor",
        [](
            const hp::Tensor<std::byte>& src,
            const hp::Buffer<std::byte>& dst,
            std::optional<uint64_t> bufferOffset,
            std::optional<uint64_t> tensorOffset,
            std::optional<uint64_t> size,
            bool unsafe
        ) {
            hp::CopyRegion region{};
            if (bufferOffset) region.bufferOffset = *bufferOffset;
            if (tensorOffset) region.tensorOffset = *tensorOffset;
            if (size) region.size = *size;
            region.unsafe = unsafe;
            return hp::retrieveTensor(src, dst, region);
        }, "src"_a, "dst"_a,
        "bufferOffset"_a.none() = nb::none(),
        "tensorOffset"_a.none() = nb::none(),
        "size"_a.none() = nb::none(),
        "unsafe"_a = false,
        "Creates a command for copying the src tensor back to the destination buffer"
        "\n\nParameters\n----------\n"
        "src: Tensor\n"
        "    Source tensor\n"
        "dst: Buffer\n"
        "    Destination buffer\n"
        "bufferOffset: None|int, default=None\n"
        "    Optional offset into the buffer in bytes\n"
        "tensorOffset: None|int, default=None\n"
        "    Optional offset into the tensor in bytes\n"
        "size: None|int, default=None\n"
        "    Amount of data to copy in bytes. If None, equals to the complete buffer\n"
        "unsafe: bool, default=False\n"
        "   Wether to omit barriers ensuring read after write ordering");
    //update tensor command
    nb::class_<hp::UpdateTensorCommand, hp::Command>(m, "UpdateTensorCommand",
            "Command for copying the src buffer into the destination tensor")
        .def("__init__",
            [](
                hp::UpdateTensorCommand* cmd,
                const hp::Buffer<std::byte>& src,
                const hp::Tensor<std::byte>& dst,
                std::optional<uint64_t> bufferOffset,
                std::optional<uint64_t> tensorOffset,
                std::optional<uint64_t> size,
                bool unsafe
            ) {
                hp::CopyRegion region{};
                if (bufferOffset) region.bufferOffset = *bufferOffset;
                if (tensorOffset) region.tensorOffset = *tensorOffset;
                if (size) region.size = *size;
                region.unsafe = unsafe;
                new (cmd) hp::UpdateTensorCommand(src, dst, region);
            }, "src"_a, "dst"_a,
            "bufferOffset"_a.none() = nb::none(),
            "tensorOffset"_a.none() = nb::none(),
            "size"_a.none() = nb::none(),
            "unsafe"_a = false);
    m.def("updateTensor",
        [](
            const hp::Buffer<std::byte>& src,
            const hp::Tensor<std::byte>& dst,
            std::optional<uint64_t> bufferOffset,
            std::optional<uint64_t> tensorOffset,
            std::optional<uint64_t> size,
            bool unsafe
        ) {
            hp::CopyRegion region{};
            if (bufferOffset) region.bufferOffset = *bufferOffset;
            if (tensorOffset) region.tensorOffset = *tensorOffset;
            if (size) region.size = *size;
            region.unsafe = unsafe;
            return hp::updateTensor(src, dst, region);
        }, "src"_a, "dst"_a,
        "bufferOffset"_a.none() = nb::none(),
        "tensorOffset"_a.none() = nb::none(),
        "size"_a.none() = nb::none(),
        "unsafe"_a = false,
        "Creates a command for copying the src buffer into the destination tensor"
        "\n\nParameters\n----------\n"
        "src: Buffer\n"
        "    Source Buffer\n"
        "dst: Tensor\n"
        "    Destination Tensor\n"
        "bufferOffset: None|int, default=None\n"
        "    Optional offset into the buffer in bytes\n"
        "tensorOffset: None|int, default=None\n"
        "    Optional offset into the tensor in bytes\n"
        "size: None|int, default=None\n"
        "    Amount of data to copy in bytes. If None, equals to the complete buffer\n"
        "unsafe: bool, default=False\n"
        "   Wether to omit barriers ensuring read after write ordering");
    //clear tensor command
    nb::class_<hp::ClearTensorCommand, hp::Command>(m, "ClearTensorCommand",
            "Command for filling a tensor with constant data over a given range")
        .def("__init__",
            [](
                hp::ClearTensorCommand* cmd,
                const hp::Tensor<std::byte>& tensor,
                std::optional<uint64_t> offset,
                std::optional<uint64_t> size,
                std::optional<uint32_t> data,
                bool unsafe
            ) {
                hp::ClearTensorCommand::Params params{};
                if (offset) params.offset = *offset;
                if (size) params.size = *size;
                if (data) params.data = *data;
                params.unsafe = unsafe;
                new (cmd) hp::ClearTensorCommand(tensor, params);
            },
            "tensor"_a,
            "offset"_a.none() = nb::none(),
            "size"_a.none() = nb::none(),
            "data"_a.none() = nb::none(),
            "unsafe"_a = false,
            "Creates a command for filling a tensor with constant data over a given range. "
            "Defaults to zeroing the complete tensor");
    m.def("clearTensor",
        [](
            const hp::Tensor<std::byte>& tensor,
            std::optional<uint64_t> offset,
            std::optional<uint64_t> size,
            std::optional<uint32_t> data,
            bool unsafe
        ) {
            hp::ClearTensorCommand::Params params{};
            if (offset) params.offset = *offset;
            if (size) params.size = *size;
            if (data) params.data = *data;
            return hp::clearTensor(tensor, params);
        },
        "tensor"_a,
        "offset"_a.none() = nb::none(),
        "size"_a.none() = nb::none(),
        "data"_a.none() = nb::none(),
        "unsafe"_a = false,
        "Creates a command for filling a tensor with constant data over a given range. "
        "Defaults to zeroing the complete tensor"
        "\n\nParameters\n----------\n"
        "tensor: Tensor\n"
        "    Tensor to be modified\n"
        "offset: None|int, default=None\n"
        "    Offset into the Tensor at which to start clearing it. "
            "Defaults to the start of the Tensor.\n"
        "size: None|int, default=None\n"
        "    Amount of bytes to clear. If None, equals to the range starting "
            "at offset until the end of the tensor\n"
        "data: None|int, default=None\n"
        "    32 bit integer used to fill the tensor. If None, uses all zeros.\n"
        "unsafe: bool, default=false\n"
        "   Wether to omit barriers ensuring read after write ordering.");
}
