#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <hephaistos/stopwatch.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerStopWatchModule(nb::module_& m) {
    nb::class_<hp::StopWatch>(m, "StopWatch",
            "Allows the measuring of elapsed time between commands execution")
        .def("__init__",
            [](hp::StopWatch* w) {
                nb::gil_scoped_release release;
                new (w) hp::StopWatch(getCurrentContext());
            },
            "Creates a new stopwatch for measuring elapsed time between commands.")
        .def("start", &hp::StopWatch::start, nb::rv_policy::reference_internal,
            "Returns the command to start the stop watch.")
        .def("stop", &hp::StopWatch::stop, nb::rv_policy::reference_internal,
            "Returns the command to stop the stop watch. Can be recorded multiple "
            "times to record up to stopCount stop times.")
        .def("reset", &hp::StopWatch::reset, "Resets the stop watch.")
        .def("getElapsedTime", &hp::StopWatch::getElapsedTime, "wait"_a = false,
            "Calculates the elapsed time between the timestamps the device recorded "
            "during its execution of the start() end stop() command in nanoseconds. "
            "If wait is true, blocks the call until both timestamps are recorded, "
            "otherwise returns NaN if they are not yet available.");
}
