#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iomanip>
#include <sstream>
#include <string>

#include <hephaistos/debug.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::string printMessage(const hp::DebugMessage& msg) {
    //Create prefix
    std::string prefix("");
    if (msg.severity & hp::DebugMessageSeverityFlagBits::VERBOSE_BIT) prefix = "[VERB]";
    if (msg.severity & hp::DebugMessageSeverityFlagBits::INFO_BIT) prefix = "[INFO]";
    if (msg.severity & hp::DebugMessageSeverityFlagBits::WARNING_BIT) prefix = "[WARN]";
    if (msg.severity & hp::DebugMessageSeverityFlagBits::ERROR_BIT) prefix = "[ERR]";

    //Create message
    std::stringstream stream;
    stream << prefix << "(" << msg.idNumber << ": " << msg.pIdName << ") " << msg.pMessage;

    //done
    return stream.str();
}

hp::DebugCallback pyCallback = nullptr;
hp::DebugCallback noneCallback = nullptr;
void pyDebugCallback(const hp::DebugMessage& msg) {
    if (pyCallback) {
        pyCallback(msg);
    }
    else {
        nb::gil_scoped_acquire gil;
        nb::print(printMessage(msg).c_str());
    }
}
void configureDebug(
    bool enablePrint,
    bool enableGPUValidation,
    bool enableSynchronizationValidation,
    bool enableThreadSafetyValidation,
    bool enableAPIValidation,
    hp::DebugCallback callback
) {
    pyCallback = std::move(callback);
    hp::configureDebug({
        .enablePrint = enablePrint,
        .enableGPUValidation = enableGPUValidation,
        .enableSynchronizationValidation = enableSynchronizationValidation,
        .enableThreadSafetyValidation = enableThreadSafetyValidation,
        .enableAPIValidation = enableAPIValidation
    }, pyDebugCallback);
}
void cleanUpDebug() {
    pyCallback = nullptr;
}

}

void registerDebugCallback(nb::module_& m) {
    nb::enum_<hp::DebugMessageSeverityFlagBits>(m, "DebugMessageSeverityFlagBits",
            "Flags indicating debug message severity")
        .value("VERBOSE", hp::DebugMessageSeverityFlagBits::VERBOSE_BIT)
        .value("INFO", hp::DebugMessageSeverityFlagBits::INFO_BIT)
        .value("WARNING", hp::DebugMessageSeverityFlagBits::WARNING_BIT)
        .value("ERROR", hp::DebugMessageSeverityFlagBits::ERROR_BIT)
        .export_values();
    
    nb::class_<hp::DebugMessage>(m, "DebugMessage",
            "Debug message")
        .def_ro("idName", &hp::DebugMessage::pIdName)
        .def_ro("idNumber", &hp::DebugMessage::idNumber)
        .def_ro("message", &hp::DebugMessage::pMessage)
        .def("__str__", &printMessage);

    m.def("isDebugAvailable", &hp::isDebugAvailable,
        "Checks whether debugging is available");
    
    m.def("configureDebug", &configureDebug,
        //nb::kw_only(), //not released yet
        "enablePrint"_a = false,
        "enableGPUValidation"_a = false,
        "enableSynchronizationValidation"_a = false,
        "enableThreadSafetyValidation"_a = false,
        "enableAPIValidation"_a = false,
        "callback"_a = noneCallback,
        "Configures the debug state");
    
    //register clean up code
    //TODO: Got this from pybind11, not sure if nanobind has something more clever
    auto atexit = nb::module_::import_("atexit");
    atexit.attr("register")(nb::cpp_function(&cleanUpDebug));
}

namespace {

std::string printDeviceFaultAddressInfo(const hp::DeviceFaultAddressInfo& info) {
    std::stringstream stream;
    stream << std::hex << std::setfill('0');
    if (info.precision <= 1) {
        stream << "address fault @ 0x" << std::setw(16) << info.address << " : ";
    }
    else {
        auto lo = info.address & ~(info.precision - 1);
        auto hi = info.address | (info.precision - 1);
        stream << "addres fault in range 0x" << std::setw(16) << lo
               << "-0x"<< std::setw(16) << hi << " : ";
    }
    switch (info.addressType) {
    case hp::DeviceFaultAddressType::None:
        stream << "None";
        break;
    case hp::DeviceFaultAddressType::ReadInvalid:
        stream << "Read Invalid";
        break;
    case hp::DeviceFaultAddressType::WriteInvalid:
        stream << "Write Invalid";
        break;
    case hp::DeviceFaultAddressType::ExecuteInvalid:
        stream << "Execute Invalid";
        break;
    case hp::DeviceFaultAddressType::InstructionPointerUnknown:
        stream << "Instruction Pointer Unknown";
        break;
    case hp::DeviceFaultAddressType::InstructionPointerInvalid:
        stream << "Instruction Pointer Invalid";
        break;
    case hp::DeviceFaultAddressType::InstructionPointerFault:
        stream << "Instruction Pointer Fault";
        break;
    default:
        stream << "Unknown (" << std::dec << static_cast<uint32_t>(info.addressType) << ")";
        break;
    }
    return stream.str();
}

std::string printDeviceVendorInfo(const hp::DeviceFaultVendorInfo& info) {
    std::stringstream stream;
    if (info.description.empty())
        stream << "(No description)";
    else
        stream << info.description;
    stream << " [" << info.code << "]";
    return stream.str();
}

bool isDeviceFaultExtSupported(uint32_t id) {
    return hp::isDeviceFaultExtensionSupported(getDevice(id));
}

}


void registerDeviceFaultExtension(nb::module_& m) {
    nb::enum_<hp::DeviceFaultAddressType>(m, "DeviceFaultAddressType",
            "Enumeration of address fault types that may have caused a device loss")
        .value("none", hp::DeviceFaultAddressType::None)
        .value("ReadInvalid", hp::DeviceFaultAddressType::ReadInvalid)
        .value("WriteInvalid", hp::DeviceFaultAddressType::WriteInvalid)
        .value("ExecuteInvalid", hp::DeviceFaultAddressType::ExecuteInvalid)
        .value("InstructionPointerUnknown", hp::DeviceFaultAddressType::InstructionPointerUnknown)
        .value("InstructionPointerInvalid", hp::DeviceFaultAddressType::InstructionPointerInvalid)
        .value("InstructionPointerFault", hp::DeviceFaultAddressType::InstructionPointerFault);
    
    nb::class_<hp::DeviceFaultAddressInfo>(m, "DeviceFaultAddressInfo",
            "Specifies memory address at which a fault occurred")
        .def(nb::init<>())
        .def_rw("addressType", &hp::DeviceFaultAddressInfo::addressType,
            "Type of memory operation that triggered the fault")
        .def_rw("address", &hp::DeviceFaultAddressInfo::address,
            "Address at which the fault occurred")
        .def_rw("precision", &hp::DeviceFaultAddressInfo::precision,
            "Precision of the reported address")
        .def("__repr__", &printDeviceFaultAddressInfo);
    nb::class_<hp::DeviceFaultVendorInfo>(m, "DeviceFaultVendorInfo",
            "Vendor specific fault information")
        .def(nb::init<>())
        .def_rw("description", &hp::DeviceFaultVendorInfo::description,
            "Human readable description of the fault")
        .def_rw("code", &hp::DeviceFaultVendorInfo::code,
            "Vendor specific fault code")
        .def_rw("data", &hp::DeviceFaultVendorInfo::data,
            "Vendor specific data associated with the fault")
        .def("__repr__", &printDeviceVendorInfo);
    nb::class_<hp::DeviceFaultInfo>(m, "DeviceFaultInfo",
            "Structure containing information about device fault")
        .def_ro("description", &hp::DeviceFaultInfo::description,
            "Human readable description of fault")
        .def_ro("addressInfo", &hp::DeviceFaultInfo::addressInfo,
            "List of address faults")
        .def_ro("vendorInfo", &hp::DeviceFaultInfo::vendorInfo,
            "List of vendor specific faults");
    
    m.def("isDeviceFaultExtensionSupported", &isDeviceFaultExtSupported,
        "id"_a, "Checks whether the given device supports reporting device fault infos");
    m.def("enableDeviceFaultExtension", [](bool force) {
            addExtension(hp::createDeviceFaultInfoExtension(), force);
        },
        nb::kw_only(), "force"_a = false,
        "Enables the device fault extension. Set `force` to `True`, if an existing "
        "context should be destroyed.");
    m.def("getDeviceFaultInfo", []() { return hp::getDeviceFaultInfo(getCurrentContext()); },
        "Queries information about the last device lost error. "
        "May only be called if such an error occurred.");
}

void registerDebugModule(nb::module_& m) {
    registerDebugCallback(m);
    registerDeviceFaultExtension(m);
}
