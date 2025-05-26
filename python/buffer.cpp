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

enum class DType : uint16_t {
    //[type] [bits] matching dlpack
    e_int8 = 0x0008,
    e_int16 = 0x0010,
    e_int32 = 0x0020,
    e_int64 = 0x0040,
    e_uint8 = 0x0108,
    e_uint16 = 0x0110,
    e_uint32 = 0x0120,
    e_uint64 = 0x0140,
    e_float16 = 0x0210,
    e_float32 = 0x0220,
    e_float64 = 0x0240
};

namespace {

constexpr auto units = std::to_array({" B", " KB", " MB", " GB"});

void printSize(size_t size, std::ostream& str) {
    int dim = 0;
    for (; dim < units.size() - 1 && size > 9216; dim++, size >>= 10);
    str << size << units[dim];
}

#define PRINT_TYPE(x) case DType::e_##x: str << #x; break;

void printType(DType dtype, std::ostream& str) {
    switch(dtype) {
    PRINT_TYPE(int8)
    PRINT_TYPE(int16)
    PRINT_TYPE(int32)
    PRINT_TYPE(int64)
    PRINT_TYPE(uint8)
    PRINT_TYPE(uint16)
    PRINT_TYPE(uint32)
    PRINT_TYPE(uint64)
    PRINT_TYPE(float16)
    PRINT_TYPE(float32)
    PRINT_TYPE(float64)
    default:
        str << "unknown";
    }
}

#undef PRINT_TYPE

nb::dlpack::dtype toDLPack(DType dtype) {
    return {
        static_cast<uint8_t>(static_cast<uint16_t>(dtype) >> 8),
        static_cast<uint8_t>(static_cast<uint16_t>(dtype) & 0xFF),
        1
    };
}
constexpr uint64_t getItemSize(DType dtype) {
    return (static_cast<uint64_t>(dtype) & 0xFF) / 8;
}
uint64_t getArraySize(std::span<const size_t> shape, nb::dlpack::dtype dtype) {
    uint64_t nbytes = dtype.bits / 8;
    for (auto s : shape) nbytes *= s;
    return nbytes;
}

using ndarray = nb::ndarray<nb::numpy, nb::device::cpu>;
using const_ndarray = nb::ndarray<nb::numpy, nb::device::cpu, nb::ro, nb::c_contig>;

std::span<const std::byte> as_span(const const_ndarray& array) {
    return { reinterpret_cast<const std::byte*>(array.data()), array.nbytes() };
}

class Buffer : public hp::Buffer<std::byte> {
public:
    ndarray numpy() const { return array; }
    
    DType dtype() const { return _dtype; }
    size_t ndim() const { return array.ndim(); }
    size_t size() const { return array.size(); }
    nb::tuple shape() const {
        nb::list result;
        for (auto i = 0; i < ndim(); ++i)
            result.append(array.shape(i));
        return nb::tuple(result);
    }

    std::string print() const {
        std::ostringstream str;
        str << "Buffer (";
        printSize(size_bytes(), str);
        str << ") of type ";
        printType(_dtype, str);
        str << "[";
        for (auto i = 0; i < array.ndim() - 1; ++i) {
            str << array.shape(i) << ",";
        }
        str << array.shape(array.ndim() - 1) << "]";
        return str.str();
    }
    
    Buffer(std::span<const size_t> shape, DType dtype)
        : hp::Buffer<std::byte>(getCurrentContext(), getArraySize(shape, toDLPack(dtype)))
        , array(getMemory().data(), shape.size(), shape.data(), {}, nullptr, toDLPack(dtype))
        , _dtype(dtype)
    {}
    ~Buffer() override = default;
    
private:
    ndarray array;
    DType _dtype;
};

template<DType DT>
class TypedBuffer : public Buffer {
public:
    TypedBuffer(std::span<const size_t> shape) : Buffer(shape, DT) {}
    ~TypedBuffer() override = default;    
};

template<DType DT>
void registerTypedBuffer(nb::module_& m, const char* name) {
    nb::class_<TypedBuffer<DT>, Buffer>(m, name)
        .def("__init__", [](TypedBuffer<DT>* b, size_t size) {
            new (b) TypedBuffer<DT>({ &size, 1 });
        }, "size"_a)
        .def("__init__", [](TypedBuffer<DT>* b, nb::typed<nb::tuple, nb::int_> shape) {
            auto ndim = shape.size();
            std::vector<size_t> sizes(ndim);
            for (auto i = 0; i < ndim; ++i)
                sizes[i] = nb::cast<size_t>(shape[i]);
            new (b) Buffer(sizes, DT);
        }, "shape"_a);
}

template<DType DT>
class TypedTensor : public hp::Tensor<std::byte> {
public:
    TypedTensor(size_t size, bool mapped)
        : hp::Tensor<std::byte>(getCurrentContext(), size, mapped)
    {}
    ~TypedTensor() override = default;
};

template<DType DT>
void registerTypedTensor(nb::module_& m, const char* name) {
    nb::class_<TypedTensor<DT>, hp::Tensor<std::byte>>(m, name)
        .def("__init__", [](TypedTensor<DT>* t, size_t size, bool mapped) {
                size *= getItemSize(DT);
                new (t) TypedTensor<DT>(size, mapped);
            }, "_size"_a, nb::kw_only(), "mapped"_a = false)
        .def("__init__", [](TypedTensor<DT>* t, nb::typed<nb::tuple, nb::int_> shape, bool mapped) {
                auto ndim = shape.size();
                auto size = getItemSize(DT);
                for (auto i = 0; i < ndim; ++i)
                    size *= nb::cast<uint64_t>(shape[i]);
                new (t) TypedTensor<DT>(size, mapped);
            }, "shape"_a, nb::kw_only(), "mapped"_a = false);
}

}

void registerBufferModule(nb::module_& m) {
    nb::enum_<DType>(m, "DType")
        .value("int8", DType::e_int8)
        .value("int16", DType::e_int16)
        .value("int32", DType::e_int32)
        .value("int64", DType::e_int64)
        .value("uint8", DType::e_uint8)
        .value("uint16", DType::e_uint16)
        .value("uint32", DType::e_uint32)
        .value("uint64", DType::e_uint64)
        .value("float16", DType::e_float16)
        .value("float32", DType::e_float32)
        .value("float64", DType::e_float64)
        .export_values();

    nb::class_<hp::Buffer<std::byte>, hp::Resource>(m, "Buffer",
            "Buffer for allocating a raw chunk of memory on the host "
            "accessible via its memory address. "
            "Useful as a base class providing more complex functionality."
            "\n\nParameters\n----------\n"
            "size: int\n"
            "    size of the buffer in bytes")
        .def("__init__", [](hp::Buffer<std::byte>* b, size_t size) {
                new (b) hp::Buffer<std::byte>(getCurrentContext(), size);
            }, "size"_a)
        .def_prop_ro("address", [](const hp::Buffer<std::byte>& b) {
                return reinterpret_cast<int64_t>(b.getMemory().data());
            },
            "The memory address of the allocated buffer.")
        .def_prop_ro("nbytes", [](const hp::Buffer<std::byte>& b) { return b.size_bytes(); },
            "The size of the buffer in bytes.")
        .def("__repr__", [](const hp::Buffer<std::byte>& b) {
            std::ostringstream str;
            str << "Buffer: ";
            printSize(b.size_bytes(), str);
            str << '\n';
            return str.str();
        });
    nb::class_<Buffer, hp::Buffer<std::byte>>(m, "NDBuffer",
            "Buffer allocating memory on the host accessible on the Python side"
            "via a numpy array.\n\n"
            "Parameters\n"
            "----------\n"
            "shape\n"
            "    Shape of the corresponding numpy array\n"
            "dtype: hephaistos::DType, default=float32\n"
            "    Data type of the corresponding numpy array.\n")
        .def("__init__", [](Buffer* b, size_t size, DType dtype) {
                new (b) Buffer({ &size, 1 }, dtype);
            }, "shape"_a, nb::kw_only(), "dtype"_a = DType::e_float32)
        .def("__init__", [](Buffer* b, nb::typed<nb::tuple, nb::int_> shape, DType dtype) {
                auto ndim = shape.size();
                std::vector<size_t> sizes(ndim);
                for (auto i = 0; i < ndim; ++i)
                    sizes[i] = nb::cast<size_t>(shape[i]);
                new (b) Buffer(sizes, dtype);
            }, "shape"_a, nb::kw_only(), "dtype"_a = DType::e_float32)
        .def("numpy", &Buffer::numpy, nb::rv_policy::reference_internal,
            "Returns numpy array using the buffer's memory")
        .def_prop_ro("dtype", &Buffer::dtype,
            "Underlying data type.")
        .def_prop_ro("ndim", &Buffer::ndim,
            "Number of dimensions of the underlying array")
        .def_prop_ro("shape", &Buffer::shape,
            "Shape of the underlying array")
        .def_prop_ro("size", &Buffer::size,
            "Returns total number of items in the buffer or array.")
        .def("__repr__", &Buffer::print);
    
    nb::class_<hp::Tensor<std::byte>, hp::Resource>(m, "Tensor",
            "Tensor allocating memory on the device.")
        .def("__init__", [](hp::Tensor<std::byte>* t, size_t size, bool mapped) {
                new (t) hp::Tensor<std::byte>(getCurrentContext(), size, mapped);
            }, "size"_a, nb::kw_only(), "mapped"_a = false,
            "Creates a new tensor of given size.\n\n"
            "Parameters\n"
            "----------\n"
            "size: int\n"
            "    Size of the tensor in bytes\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map the tensor to host address space.")
        .def("__init__", [](hp::Tensor<std::byte>* t, size_t nbytes, uint64_t addr, bool mapped) {
                new (t) hp::Tensor<std::byte>(
                    getCurrentContext(),
                    { reinterpret_cast<const std::byte*>(addr), nbytes },
                    mapped
                );
            }, "size"_a, nb::kw_only(), "addr"_a, "mapped"_a = false,
            "Creates a new tensor and fills it with the provided data"
            "\n\nParameters\n----------\n"
            "addr: int\n"
            "    Address of the data used to fill the tensor\n"
            "n: int\n"
            "    Number of bytes to copy from addr\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map memory to host address space")
        .def("__init__", [](hp::Tensor<std::byte>* t, const const_ndarray& array, bool mapped) {
                new (t) hp::Tensor<std::byte>(
                    getCurrentContext(),
                    { reinterpret_cast<const std::byte*>(array.data()), array.nbytes() },
                    mapped
                );
            }, "array"_a, nb::kw_only(), "mapped"_a = false,
            "Creates a new tensor and fills it with the provided data"
            "\n\nParameters\n----------\n"
            "array: NDArray\n"
            "    Numpy array containing the data used to fill the tensor. "
                "Its type must match the tensor's.\n"
            "mapped: bool, default=False\n"
            "    If True, tries to map memory to host address space")
        .def_prop_ro("address", [](const hp::Tensor<std::byte>& t) { return t.address(); },
            "The device address of this tensor.")
        .def_prop_ro("isMapped", [](const hp::Tensor<std::byte>& t) { return t.isMapped(); },
            "True, if the underlying memory is writable by the CPU.")
        .def_prop_ro("memory", [](const hp::Tensor<std::byte>& t)
                { return reinterpret_cast<intptr_t>(t.getMemory().data()); },
            "Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.")
        .def_prop_ro("nbytes", [](const hp::Tensor<std::byte>& t) { return t.size_bytes(); },
            "The size of the tensor in bytes.")
        .def_prop_ro("isNonCoherent", [](const hp::Tensor<std::byte>& t) { return t.isNonCoherent(); },
            "Wether calls to flush() and invalidate() are necessary to make changes"
            "in mapped memory between devices and host available")
        .def("update",
            [](hp::Tensor<std::byte>& t, uint64_t ptr, size_t n, uint64_t offset) {
                t.update({ reinterpret_cast<std::byte*>(ptr), n }, offset);
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
            [](hp::Tensor<std::byte>& t, uint64_t offset, std::optional<uint64_t> size) {
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
            [](hp::Tensor<std::byte>& t, uint64_t ptr, size_t n, uint64_t offset) {
                t.retrieve({ reinterpret_cast<std::byte*>(ptr), n }, offset);
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
            [](hp::Tensor<std::byte>& t, uint64_t offset, std::optional<uint64_t> size) {
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
        .def("bindParameter", [](const hp::Tensor<std::byte>& t, hp::Program& p, uint32_t b)
            { t.bindParameter(p.getBinding(b)); }, "program"_a, "binding"_a,
            "Binds the tensor to the program at the given binding")
        .def("bindParameter", [](const hp::Tensor<std::byte>& t, hp::Program& p, std::string_view b)
            { t.bindParameter(p.getBinding(b)); }, "program"_a, "binding"_a,
            "Binds the tensor to the program at the given binding")
        .def("__repr__", [](const hp::Tensor<std::byte>& t) {
            std::ostringstream str;
            str << "Tensor (";
            printSize(t.size_bytes(), str);
            str << ") at 0x" << std::uppercase << std::hex << t.address();
            if (t.isMapped())
                str << " (mapped)\n";
            else
                str << '\n';
            return str.str();
        });

    //Register typed buffers
    registerTypedBuffer<DType::e_int8>(m, "CharBuffer");
    registerTypedBuffer<DType::e_int16>(m, "ShortBuffer");
    registerTypedBuffer<DType::e_int32>(m, "IntBuffer");
    registerTypedBuffer<DType::e_int64>(m, "LongBuffer");
    registerTypedBuffer<DType::e_uint8>(m, "ByteBuffer");
    registerTypedBuffer<DType::e_uint16>(m, "UnsignedShortBuffer");
    registerTypedBuffer<DType::e_uint32>(m, "UnsignedIntBuffer");
    registerTypedBuffer<DType::e_uint64>(m, "UnsignedLongBuffer");
    registerTypedBuffer<DType::e_float16>(m, "HalfBuffer");
    registerTypedBuffer<DType::e_float32>(m, "FloatBuffer");
    registerTypedBuffer<DType::e_float64>(m, "DoubleBuffer");

    //Register typed buffers
    registerTypedTensor<DType::e_int8>(m, "CharTensor");
    registerTypedTensor<DType::e_int16>(m, "ShortTensor");
    registerTypedTensor<DType::e_int32>(m, "IntTensor");
    registerTypedTensor<DType::e_int64>(m, "LongTensor");
    registerTypedTensor<DType::e_uint8>(m, "ByteTensor");
    registerTypedTensor<DType::e_uint16>(m, "UnsignedShortTensor");
    registerTypedTensor<DType::e_uint32>(m, "UnsignedIntTensor");
    registerTypedTensor<DType::e_uint64>(m, "UnsignedLongTensor");
    registerTypedTensor<DType::e_float16>(m, "HalfTensor");
    registerTypedTensor<DType::e_float32>(m, "FloatTensor");
    registerTypedTensor<DType::e_float64>(m, "DoubleTensor");

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
