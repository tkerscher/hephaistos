#include <nanobind/nanobind.h>

#include <hephaistos/command.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

// I really need to wrap my head around how nanobind handles overloading...

class PyTimeline : public hp::Timeline {
public:
    void wait(uint64_t value) {
        hp::Timeline::waitValue(value);
    }
    bool waitTimeout(uint64_t value, uint64_t timeout) {
        return hp::Timeline::waitValue(value, timeout);
    }

    PyTimeline(uint64_t initialValue = 0)
        : hp::Timeline(getCurrentContext(), initialValue)
    {}
    virtual ~PyTimeline() = default;
};

void registerCommandModule(nb::module_& m) {
    nb::class_<CommandHandle>(m, "CommandHandle");

    nb::class_<PyTimeline>(m, "Timeline")
        .def(nb::init<>())
        .def(nb::init<uint64_t>())
        .def_prop_rw("value",
            [](const PyTimeline& t) { return t.getValue(); },
            [](PyTimeline& t, uint64_t v) { t.setValue(v); },
            "Returns or sets the current value of the timeline. Note that the value can only increase.")
        .def("wait", &PyTimeline::wait, "value"_a,
            "Waits for the timeline to reach the given value.")
        .def("waitTimeout", &PyTimeline::waitTimeout, "value"_a, "ns"_a,
            "Waits for the timeline to reach the given value for a certain amount. Returns True if the value was reached and False if it timed out.");

    nb::class_<hp::SequenceBuilder>(m, "SequenceBuilder")
        .def("__init__", [](hp::SequenceBuilder* sb, PyTimeline& t) { new (sb) hp::SequenceBuilder(t); }, "timeline"_a)
        .def("__init__", [](hp::SequenceBuilder* sb, PyTimeline& t, uint64_t v) { new (sb) hp::SequenceBuilder(t, v); }, "timeline"_a, "startValue"_a)
        .def("And", [](hp::SequenceBuilder& sb, CommandHandle& h) -> hp::SequenceBuilder& { return sb.And(h.command); }, "cmd"_a,
            "Issues the command to run parallel in the current step.", nb::rv_policy::reference_internal)
        .def("Then", [](hp::SequenceBuilder& sb, CommandHandle& h) -> hp::SequenceBuilder& { return sb.Then(h.command); }, "cmd"_a,
            "Issues a new step to execute after waiting for the previous one to finish.", nb::rv_policy::reference_internal)
        .def("Wait", &hp::SequenceBuilder::WaitFor, "value"_a,
            "Issues a new step to execute after the timeline reaches the given value.", nb::rv_policy::reference_internal)
        .def("Submit", &hp::SequenceBuilder::Submit, "Submits the recorded steps as a single batch to the GPU.");
    
    m.def("beginSequence", [](PyTimeline& t, uint64_t v) { return hp::beginSequence(t, v); },
        "timeline"_a, "startValue"_a = 0, "Starts a new sequence.", nb::rv_policy::move);
}
