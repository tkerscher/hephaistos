#include <nanobind/nanobind.h>

#include <stdexcept>

#include <hephaistos/atomic.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

enum class Atomics {
    BufferInt64,
    BufferFloat16LoadStore,
    BufferFloat16Add,
    BufferFloat16MinMax,
    BufferFloat32LoadStore,
    BufferFloat32Add,
    BufferFloat32MinMax,
    BufferFloat64LoadStore,
    BufferFloat64Add,
    BufferFloat64MinMax,
    SharedInt64,
    SharedFloat16LoadStore,
    SharedFloat16Add,
    SharedFloat16MinMax,
    SharedFloat32LoadStore,
    SharedFloat32Add,
    SharedFloat32MinMax,
    SharedFloat64LoadStore,
    SharedFloat64Add,
    SharedFloat64MinMax,
    ImageInt64,
    ImageFloat32LoadStore,
    ImageFloat32Add,
    ImageFloat32MinMax
};

nb::set castAtomicPropertiesToSet(const hp::AtomicsProperties& props) {
    nb::set result{};

    if (props.bufferInt64Atomics) result.add(Atomics::BufferInt64);

    if (props.bufferFloat16AtomicLoadStore) result.add(Atomics::BufferFloat16LoadStore);
    if (props.bufferFloat16AtomicAdd) result.add(Atomics::BufferFloat16Add);
    if (props.bufferFloat16AtomicMinMax) result.add(Atomics::BufferFloat16MinMax);

    if (props.bufferFloat32AtomicLoadStore) result.add(Atomics::BufferFloat32LoadStore);
    if (props.bufferFloat32AtomicAdd) result.add(Atomics::BufferFloat32Add);
    if (props.bufferFloat32AtomicMinMax) result.add(Atomics::BufferFloat32MinMax);

    if (props.bufferFloat64Atomics) result.add(Atomics::BufferFloat64LoadStore);
    if (props.bufferFloat64AtomicAdd) result.add(Atomics::BufferFloat64Add);
    if (props.bufferFloat64AtomicMinMax) result.add(Atomics::BufferFloat64MinMax);

    if (props.sharedInt64Atomics) result.add(Atomics::SharedInt64);

    if (props.sharedFloat16AtomicLoadStore) result.add(Atomics::SharedFloat16LoadStore);
    if (props.sharedFloat16AtomicAdd) result.add(Atomics::SharedFloat16Add);
    if (props.sharedFloat16AtomicMinMax) result.add(Atomics::SharedFloat16MinMax);

    if (props.sharedFloat32AtomicLoadStore) result.add(Atomics::SharedFloat32LoadStore);
    if (props.sharedFloat32AtomicAdd) result.add(Atomics::SharedFloat32Add);
    if (props.sharedFloat32AtomicMinMax) result.add(Atomics::SharedFloat32MinMax);

    if (props.sharedFloat64Atomics) result.add(Atomics::SharedFloat64LoadStore);
    if (props.sharedFloat64AtomicAdd) result.add(Atomics::SharedFloat64Add);
    if (props.sharedFloat64AtomicMinMax) result.add(Atomics::SharedFloat64MinMax);

    if (props.imageInt64Atomics) result.add(Atomics::ImageInt64);

    if (props.imageFloat32AtomicLoadStore) result.add(Atomics::ImageFloat32LoadStore);
    if (props.imageFloat32AtomicAdd) result.add(Atomics::ImageFloat32Add);
    if (props.imageFloat32AtomicMinMax) result.add(Atomics::ImageFloat32MinMax);

    return result;
}

hp::AtomicsProperties castSetToAtomicProperties(const nb::set& set) {
    hp::AtomicsProperties props{};
    for (nb::handle h : set) {
        auto flag = nb::cast<Atomics>(h);
        switch (flag)
        {
        case Atomics::BufferInt64:
            props.bufferInt64Atomics = true;
            break;
        case Atomics::BufferFloat16LoadStore:
            props.bufferFloat16AtomicLoadStore = true;
            break;
        case Atomics::BufferFloat16Add:
            props.bufferFloat16AtomicAdd = true;
            break;
        case Atomics::BufferFloat16MinMax:
            props.bufferFloat16AtomicMinMax = true;
            break;
        case Atomics::BufferFloat32LoadStore:
            props.bufferFloat32AtomicLoadStore = true;
            break;
        case Atomics::BufferFloat32Add:
            props.bufferFloat32AtomicAdd = true;
            break;
        case Atomics::BufferFloat32MinMax:
            props.bufferFloat32AtomicMinMax = true;
            break;
        case Atomics::BufferFloat64LoadStore:
            props.bufferFloat64Atomics = true;
            break;
        case Atomics::BufferFloat64Add:
            props.bufferFloat64AtomicAdd = true;
            break;
        case Atomics::BufferFloat64MinMax:
            props.bufferFloat64AtomicMinMax = true;
            break;
        case Atomics::SharedInt64:
            props.sharedInt64Atomics = true;
            break;
        case Atomics::SharedFloat16LoadStore:
            props.sharedFloat16AtomicLoadStore = true;
            break;
        case Atomics::SharedFloat16Add:
            props.sharedFloat16AtomicAdd = true;
            break;
        case Atomics::SharedFloat16MinMax:
            props.sharedFloat16AtomicMinMax = true;
            break;
        case Atomics::SharedFloat32LoadStore:
            props.sharedFloat32AtomicLoadStore = true;
            break;
        case Atomics::SharedFloat32Add:
            props.sharedFloat32AtomicAdd = true;
            break;
        case Atomics::SharedFloat32MinMax:
            props.sharedFloat32AtomicMinMax = true;
            break;
        case Atomics::SharedFloat64LoadStore:
            props.sharedFloat64Atomics = true;
            break;
        case Atomics::SharedFloat64Add:
            props.sharedFloat64AtomicAdd = true;
            break;
        case Atomics::SharedFloat64MinMax:
            props.sharedFloat64AtomicMinMax = true;
            break;
        case Atomics::ImageInt64:
            props.imageInt64Atomics = true;
            break;
        case Atomics::ImageFloat32LoadStore:
            props.imageFloat32AtomicLoadStore = true;
            break;
        case Atomics::ImageFloat32Add:
            props.imageFloat32AtomicAdd = true;
            break;
        case Atomics::ImageFloat32MinMax:
            props.imageFloat32AtomicMinMax = true;
            break;
        default:
            throw std::logic_error("Unexpected atomic flag!");
        }
    }
    return props;
}

nb::set getSupportedAtomics(uint32_t id) {
    auto caps = hp::getAtomicsProperties(getDevice(id));
    return castAtomicPropertiesToSet(caps);
}

nb::set getEnabledAtomics() {
    auto& caps = hp::getEnabledAtomics(getCurrentContext());
    return castAtomicPropertiesToSet(caps);
}

void enableAtomics(const nb::set& set, bool force) {
    auto props = castSetToAtomicProperties(set);
    addExtension(hp::createAtomicsExtension(props), force);
}

}

void registerAtomicModule(nb::module_& m) {
    nb::enum_<Atomics>(m, "Atomics", "Enumeration of atomic capabilities")
        .value("BufferInt64", Atomics::BufferInt64,
            "Atomic operations using 64 bit integers stored in buffers.")
        .value("BufferFloat16LoadStore", Atomics::BufferFloat16LoadStore,
            "Atomic load, store and exchange using 16 bit floats stored in buffers.")
        .value("BufferFloat16Add", Atomics::BufferFloat16Add,
            "Atomic add using 16 bit floats stored in buffers.")
        .value("BufferFloat16MinMax", Atomics::BufferFloat16MinMax,
            "Atomic min and max using 16 bit floats stored in buffers.")
        .value("BufferFloat32LoadStore", Atomics::BufferFloat32LoadStore,
            "Atomic load, store and exchange using 32 bit floats stored in buffers.")
        .value("BufferFloat32Add", Atomics::BufferFloat32Add,
            "Atomic add using 32 bit floats stored in buffers.")
        .value("BufferFloat32MinMax", Atomics::BufferFloat32MinMax,
            "Atomic min and max using 32 bit floats stored in buffers.")
        .value("BufferFloat64LoadStore", Atomics::BufferFloat64LoadStore,
            "Atomic load, store and exchange using 64 bit floats stored in buffers.")
        .value("BufferFloat64Add", Atomics::BufferFloat64Add,
            "Atomic add using 64 bit floats stored in buffers.")
        .value("BufferFloat64MinMax", Atomics::BufferFloat64MinMax,
            "Atomic min and max using 64 bit floats stored in buffers.")
        .value("SharedInt64", Atomics::SharedInt64,
            "Atomic operations using 64 bit integers stored in shared memory.")
        .value("SharedFloat16LoadStore", Atomics::SharedFloat16LoadStore,
            "Atomic load, store and exchange using 16 bit floats stored in shared memory.")
        .value("SharedFloat16Add", Atomics::SharedFloat16Add,
            "Atomic add using 16 bit floats stored in shared memory.")
        .value("SharedFloat16MinMax", Atomics::SharedFloat16MinMax,
            "Atomic min and max using 16 bit floats stored in shared memory.")
        .value("SharedFloat32LoadStore", Atomics::SharedFloat32LoadStore,
            "Atomic load, store and exchange using 32 bit floats stored in shared memory.")
        .value("SharedFloat32Add", Atomics::SharedFloat32Add,
            "Atomic add using 32 bit floats stored in shared memory.")
        .value("SharedFloat32MinMax", Atomics::SharedFloat32MinMax,
            "Atomic min and max using 32 bit floats stored in shared memory.")
        .value("SharedFloat64LoadStore", Atomics::SharedFloat64LoadStore,
            "Atomic load, store and exchange using 64 bit floats stored in shared memory.")
        .value("SharedFloat64Add", Atomics::SharedFloat64Add,
            "Atomic add using 64 bit floats stored in shared memory.")
        .value("SharedFloat64MinMax", Atomics::SharedFloat64MinMax,
            "Atomic min and max using 64 bit floats stored in shared memory.")
        .value("ImageInt64", Atomics::ImageInt64,
            "Atomic operations using 64 bit integers stored in images.")
        .value("ImageFloat32LoadStore", Atomics::ImageFloat32LoadStore,
            "Atomic load, store and exchange using 32 bit floats stored in images.")
        .value("ImageFloat32Add", Atomics::ImageFloat32Add,
            "Atomic add using 32 bit floats stored in images.")
        .value("ImageFloat32MinMax", Atomics::ImageFloat32MinMax,
            "Atomic min and max using 32 bit floats stored in images.");

    m.def("getSupportedAtomics", &getSupportedAtomics, "id"_a,
        nb::sig("def getSupportedAtomics(id: int) -> set[Atomics]"),
        "Returns a set of all supported atomic capabilites supported by the "
        "device specified by its id.");
    m.def("getEnabledAtomics", &getEnabledAtomics,
        nb::sig("def getEnabledAtomics() -> set[Atomics]"),
        "Returns the set of enabled atomic capabilities in the current context.");
    m.def("enableAtomics", &enableAtomics,
        "atomics"_a, nb::kw_only(), "force"_a = false,
        nb::sig("def enableAtomics(atomics: set[Atomics], *, force: bool = False) -> None"),
        "Enables the atomic capabilites specified in the given set by their name. "
        "Set `force=True` if an existing context should be destroyed.");
}
