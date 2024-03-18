#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <sstream>
#include <string>

#include <hephaistos/debug.hpp>

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

void registerDebugModule(nb::module_& m) {
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
