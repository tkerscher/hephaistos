#include <nanobind/nanobind.h>

#include <stdexcept>

#include <hephaistos/command.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerCommandModule(nb::module_& m) {
    nb::class_<hp::Command>(m, "Command");

    nb::class_<hp::Subroutine>(m, "Subroutine")
        .def_prop_ro("simultaneousUse",
            [](const hp::Subroutine& s) -> bool { return s.simultaneousUse(); },
            "True, if the subroutine can be used simultaneous");
    m.def("createSubroutine", [](nb::list list, bool simultaneous) -> hp::Subroutine {
        hp::SubroutineBuilder builder(getCurrentContext(), simultaneous);
        for (nb::handle h : list) {
            builder.addCommand(nb::cast<const hp::Command&>(h));
        }
        return builder.finish();
    }, "list"_a, "simultaneous"_a = false,
    "creates a subroutine from the list of commands");

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
        .def("And", [](hp::SequenceBuilder& sb, const hp::Subroutine& s) -> hp::SequenceBuilder& { return sb.And(s); }, "subroutine"_a,
            "Issues the subroutine to run parallel in the current step.", nb::rv_policy::reference_internal)
        .def("AndList", [](hp::SequenceBuilder& sb, nb::list list) -> hp::SequenceBuilder&
            {
                for (nb::handle h : list) {
                    if (nb::isinstance<hp::Command>(h)) {
                        sb.And(nb::cast<const hp::Command&>(h));
                    }
                    else if (nb::isinstance<hp::Subroutine>(h)) {
                        sb.And(nb::cast<const hp::Subroutine&>(h));
                    }
                    else {
                        //conversion failed
                        throw std::invalid_argument("Entries in the list must either be subroutines or derive from Command!");
                    }                    
                }
                //done
                return sb;
            }, "list"_a, nb::rv_policy::reference_internal,
            "Issues each element of the list to run parallel in the current step")
        .def("Then", [](hp::SequenceBuilder& sb, const hp::Command& h) -> hp::SequenceBuilder& { return sb.Then(h); }, "cmd"_a,
            "Issues a new step to execute after waiting for the previous one to finish.", nb::rv_policy::reference_internal)
        .def("Then", [](hp::SequenceBuilder& sb, const hp::Subroutine& s) -> hp::SequenceBuilder& { return sb.Then(s); }, "subroutine"_a,
            "Issues a new step to execute after waiting for the previous one to finish.", nb::rv_policy::reference_internal)
        .def("NextStep", &hp::SequenceBuilder::NextStep, nb::rv_policy::reference_internal,
            "Issues a new step. Following calls to And are ensured to run after previous ones finished.")
        .def("WaitFor", &hp::SequenceBuilder::WaitFor, "value"_a,
            "Issues a new step to execute after the timeline reaches the given value.", nb::rv_policy::reference_internal)
        .def("Submit", &hp::SequenceBuilder::Submit, "Submits the recorded steps as a single batch to the GPU.");
    
    m.def("beginSequence", [](hp::Timeline& t, uint64_t v) { return hp::beginSequence(t, v); },
        "timeline"_a, "startValue"_a = 0, "Starts a new sequence.", nb::rv_policy::move);
    m.def("beginSequence", []() { return hp::beginSequence(getCurrentContext()); },
        "Starts a new sequence.", nb::rv_policy::move);
    
    m.def("execute", [](const hp::Command& cmd) { hp::execute(getCurrentContext(), cmd); },
        "cmd"_a, "Runs the given command synchronous.");
    m.def("execute", [](const hp::Subroutine& sub) { hp::execute(getCurrentContext(), sub); },
        "sub"_a, "Runs the given subroutine synchronous.");
    m.def("executeList", [](nb::list list)
        {
            hp::execute(getCurrentContext(), [&list](hp::vulkan::Command& cmd) {
                for (nb::handle h : list) {
                    nb::cast<const hp::Command&>(h).record(cmd);
                }
            });
        }, "list"_a, "Runs the given of list commands synchronous");
}
