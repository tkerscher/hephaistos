#include <nanobind/nanobind.h>

#include <hephaistos/command.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

nb::class_<hp::Command> commandClass;

void registerCommandModule(nb::module_& m) {
    commandClass = nb::class_<hp::Command>(m, "Command");

    nb::class_<hp::Timeline>(m, "Timeline")
        .def("__init__", [](hp::Timeline* t) { new (t) hp::Timeline(getCurrentContext()); })
        .def("__init__", [](hp::Timeline* t, uint64_t v) { new (t) hp::Timeline(getCurrentContext(), v); }, "initialValue"_a)
        .def_prop_rw("value",
            [](const hp::Timeline& t) { return t.getValue(); },
            [](hp::Timeline& t, uint64_t v) { t.setValue(v); },
            "Returns or sets the current value of the timeline. Note that the value can only increase.")
        .def("wait", [](const hp::Timeline& t, uint64_t v) { t.waitValue(v); }, "value"_a,
            "Waits for the timeline to reach the given value.")
        .def("waitTimeout", [](const hp::Timeline& t, uint64_t v, uint64_t d) { return t.waitValue(v, d); }, "value"_a, "ns"_a,
            "Waits for the timeline to reach the given value for a certain amount. Returns True if the value was reached and False if it timed out.");

    nb::class_<hp::Submission>(m, "Submission")
        .def_prop_ro("timeline", [](const hp::Submission& s) -> const hp::Timeline& { return s.getTimeline(); },
            "The timeline this submission was issued with.")
        .def_prop_ro("finalStep", [](const hp::Submission& s) { return s.getFinalStep(); },
            "The value the timeline will reach when the submission finishes.")
        .def("wait", [](const hp::Submission& s) { s.wait(); },
            "Blocks the caller until the submission finishes.")
        .def("waitTimeout", [](const hp::Submission& s, uint64_t t) -> bool { return s.wait(t); }, "ns"_a,
            "Blocks the caller until the submission finished or the specified time elapsed. Returns True in the first, False in the second case.");

    nb::class_<hp::SequenceBuilder>(m, "SequenceBuilder")
        .def("__init__", [](hp::SequenceBuilder* sb, hp::Timeline& t) { new (sb) hp::SequenceBuilder(t); }, "timeline"_a)
        .def("__init__", [](hp::SequenceBuilder* sb, hp::Timeline& t, uint64_t v) { new (sb) hp::SequenceBuilder(t, v); }, "timeline"_a, "startValue"_a)
        .def("And", [](hp::SequenceBuilder& sb, const hp::Command& h) -> hp::SequenceBuilder& { return sb.And(h); }, "cmd"_a,
            "Issues the command to run parallel in the current step.", nb::rv_policy::reference_internal)
        .def("Then", [](hp::SequenceBuilder& sb, const hp::Command& h) -> hp::SequenceBuilder& { return sb.Then(h); }, "cmd"_a,
            "Issues a new step to execute after waiting for the previous one to finish.", nb::rv_policy::reference_internal)
        .def("Wait", &hp::SequenceBuilder::WaitFor, "value"_a,
            "Issues a new step to execute after the timeline reaches the given value.", nb::rv_policy::reference_internal)
        .def("Submit", &hp::SequenceBuilder::Submit, "Submits the recorded steps as a single batch to the GPU.");
    
    m.def("beginSequence", [](hp::Timeline& t, uint64_t v) { return hp::beginSequence(t, v); },
        "timeline"_a, "startValue"_a = 0, "Starts a new sequence.", nb::rv_policy::move);
}
