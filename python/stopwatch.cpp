#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <hephaistos/stopwatch.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerStopWatchModule(nb::module_& m) {
    nb::class_<hp::StopWatch>(m, "StopWatch",
            "Allows the measuring of elapsed time between commands execution"
            "\n\nParameters\n----------\n"
            "stops: int\n"
            "    Number of stops to be recorded")
        .def("__init__",
            [](hp::StopWatch* w, uint32_t stops) {
                nb::gil_scoped_release release;
                new (w) hp::StopWatch(getCurrentContext(), stops);
            },
            "stops"_a = 1,
            "Creates a new stopwatch for measuring elapsed time between commands.")
        .def_prop_ro("stopCount",
            [](const hp::StopWatch& w) -> uint32_t { return w.getStopCount(); },
            "The number of stops that can be recorded.")
        .def("start", &hp::StopWatch::start, nb::rv_policy::reference_internal,
            "Returns the command to start the stop watch.")
        .def("stop", &hp::StopWatch::stop, nb::rv_policy::reference_internal,
            "Returns the command to stop the stop watch. Can be recorded multiple "
            "times to record up to stopCount stop times.")
        .def("reset", &hp::StopWatch::reset, "Resets the stop watch.")
        .def("getTimeStamps", &hp::StopWatch::getTimeStamps, "wait"_a = false,
            "Retrieves the recorded timestamps from the device. If wait = True, "
            "blocks the caller until all timestamps are available, otherwise "
            "unavailable ones are denoted as NaN.");
}
