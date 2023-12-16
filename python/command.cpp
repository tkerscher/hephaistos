#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <sstream>
#include <stdexcept>
#include <variant>

#include <hephaistos/command.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerCommandModule(nb::module_& m) {
    nb::class_<hp::Command>(m, "Command",
        "Base class for commands running on the device. "
        "Execution happens asynchronous after being submitted.");

    nb::class_<hp::Subroutine>(m, "Subroutine",
            "Subroutines are reusable sequences of commands that can be "
            "submitted multiple times to the device. "
            "Recording sequence of commands require non negligible CPU time "
            "and may be amortized by reusing sequences via Subroutines.")
        .def_prop_ro("simultaneousUse",
            [](const hp::Subroutine& s) -> bool { return s.simultaneousUse(); },
            "True, if the subroutine can be used simultaneous");
    m.def("createSubroutine",
        [](nb::list list, bool simultaneous) -> hp::Subroutine {
            //try to minimize holding of GIL
            // -> collect commands before recording
            std::vector<const hp::Command*> commands(list.size());
            auto i = 0u;
            for (nb::handle h : list)
                commands[i++] = nb::cast<const hp::Command*>(h);

            //release GIL
            nb::gil_scoped_release release;
            //build subroutine
            hp::SubroutineBuilder builder(getCurrentContext(), simultaneous);
            for (auto c : commands)
                builder.addCommand(*c);
            return builder.finish();
        }, "commands"_a, "simultaneous"_a = false,
        "creates a subroutine from the list of commands"
        "\n\nParameters\n----------\n"
        "commands: Command[]\n"
        "    Sequence of commands the Subroutine consists of\n"
        "simultaneous: bool, default=False\n"
        "    True, if the subroutine can be submitted while a previous submission\n"
        "    has not yet finished. Disobeying this requirement results in undefined\n"
        "    behavior.");

    nb::class_<hp::Timeline>(m, "Timeline",
            "Timeline managing the execution of code and commands using an "
            "increasing internal counter. Both GPU and CPU can wait and/or "
            "increase this counter thus creating a synchronization between "
            "them or among themselves. "
            "The current value of the counter can be queried allowing an "
            "asynchronous method of reporting the progress."
            "\n\nParameters\n----------\n"
            "value: int, default=0\n"
            "    Initial value of the internal counter\n")
        .def("__init__", [](hp::Timeline* t) { new (t) hp::Timeline(getCurrentContext()); })
        .def("__init__", [](hp::Timeline* t, uint64_t v)
            { new (t) hp::Timeline(getCurrentContext(), v); }, "initialValue"_a)
        .def_prop_ro("id", [](const hp::Timeline& t) { return t.getId(); },
            "Id of this timeline")
        .def_prop_rw("value",
            [](const hp::Timeline& t) { return t.getValue(); },
            [](hp::Timeline& t, uint64_t v) { t.setValue(v); },
            "Returns or sets the current value of the timeline. "
            "Note that the value can only increase. "
            "Disobeying this requirement results in undefined behavior.")
        .def("wait", [](const hp::Timeline& t, uint64_t v) {
                nb::gil_scoped_release release;
                t.waitValue(v);
            }, "value"_a,
            "Waits for the timeline to reach the given value.")
        .def("waitTimeout", [](const hp::Timeline& t, uint64_t v, uint64_t d) {
                nb::gil_scoped_release release;
                return t.waitValue(v, d);
            }, "value"_a, "timeout"_a,
            "Waits for the timeline to reach the given value for a certain amount. "
            "Returns True if the value was reached and False if it timed out."
            "\n\nParameters\n----------\n"
            "value: int\n"
            "    Value to wait for\n"
            "timeout: int\n"
            "    Time in nanoseconds to wait. May be rounded to the closest "
                "internal precision of the device clock.")
        .def("__repr__", [](const hp::Timeline& t) -> std::string {
            std::ostringstream str;
            str << "Timeline (ID: 0x" << std::hex << t.getId() << ")\n";
            return str.str();
        });

    nb::class_<hp::Submission>(m, "Submission",
            "Submissions are issued after work has been submitted to the device "
            "and can be used to wait for its completion.")
        .def_prop_ro("timeline", [](const hp::Submission& s) -> const hp::Timeline& { return s.getTimeline(); },
            "The timeline this submission was issued with.")
        .def_prop_ro("finalStep", [](const hp::Submission& s) { return s.getFinalStep(); },
            "The value the timeline will reach when the submission finishes.")
        .def_prop_ro("forgettable", [](const hp::Submission& s){ return s.forgettable(); },
            "True, if the Submission can be discarded safely, i.e. fire and forget.")
        .def("wait", [](const hp::Submission& s) {
                nb::gil_scoped_release release;
                s.wait();
            }, "Blocks the caller until the submission finishes.")
        .def("waitTimeout", [](const hp::Submission& s, uint64_t t) -> bool {
                nb::gil_scoped_release release;
                return s.wait(t);
            }, "ns"_a,
            "Blocks the caller until the submission finished or the specified time elapsed. "
            "Returns True in the first, False in the second case.");

    nb::class_<hp::SequenceBuilder>(m, "SequenceBuilder",
            "Builder class for recording a sequence of commands and subroutines "
            "and submitting it to the device for execution, which happens "
            "asynchronous, i.e. submit() returns before the recorded work "
            "finishes.")
        .def("__init__",
            [](hp::SequenceBuilder* sb, hp::Timeline& t)
                { new (sb) hp::SequenceBuilder(t); },
            "timeline"_a,
            "Creates a new SequenceBuilder."
            "\n\nParameters\n----------\n"
            "timeline: Timeline\n"
            "    Timeline to use for orchestrating commands and subroutines")
        .def("__init__",
            [](hp::SequenceBuilder* sb, hp::Timeline& t, uint64_t v)
                { new (sb) hp::SequenceBuilder(t, v); },
            "timeline"_a, "startValue"_a,
            "Creates a new SequenceBuilder."
            "\n\nParameters\n----------\n"
            "timeline: Timeline\n"
            "    Timeline to use for orchestrating commands and subroutines\n"
            "startValue: int\n"
            "    Counter value to wait for on the timeline to start with")
        .def("And", [](hp::SequenceBuilder& sb, const hp::Command& h) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.And(h);
            }, "cmd"_a, "Issues the command to run parallel in the current step.",
            nb::rv_policy::reference_internal)
        .def("And", [](hp::SequenceBuilder& sb, const hp::Subroutine& s) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.And(s);
            }, "subroutine"_a, "Issues the subroutine to run parallel in the current step.",
            nb::rv_policy::reference_internal)
        .def("AndList", [](hp::SequenceBuilder& sb, nb::list list) -> hp::SequenceBuilder&
            {
                //try to minimize holding of GIL
                // -> collect commands before recording
                std::vector<std::variant<const hp::Command*, const hp::Subroutine*>> elements(list.size());
                auto i = 0u;
                for (nb::handle h : list) {
                    if (nb::isinstance<hp::Command>(h)) {
                        elements[i++] = nb::cast<const hp::Command*>(h);
                    }
                    else if (nb::isinstance<hp::Subroutine>(h)) {
                        elements[i++] = nb::cast<const hp::Subroutine*>(h);
                    }
                    else {
                        //conversion failed
                        throw std::invalid_argument(
                            "Entries in the list must either be subroutines or derive from Command!");
                    } 
                }

                //release GIL
                nb::gil_scoped_release release;
                struct Visitor{
                    hp::SequenceBuilder& sb;

                    void operator() (const hp::Command* c) { sb.And(*c); }
                    void operator() (const hp::Subroutine* s) { sb.And(*s); }
                } visitor{ sb };
                for (auto& e : elements)
                    std::visit(visitor, e);
                
                //done
                return sb;
            }, "list"_a, nb::rv_policy::reference_internal,
            "Issues each element of the list to run parallel in the current step")
        .def("Then", [](hp::SequenceBuilder& sb, const hp::Command& c) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.Then(c);
            }, "cmd"_a, nb::rv_policy::reference_internal,
            "Issues a new step to execute after waiting for the previous one to finish.")
        .def("Then", [](hp::SequenceBuilder& sb, const hp::Subroutine& s) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.Then(s);
            }, "subroutine"_a, nb::rv_policy::reference_internal,
            "Issues a new step to execute after waiting for the previous one to finish.")
        .def("NextStep", [](hp::SequenceBuilder& sb) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.NextStep();
            }, nb::rv_policy::reference_internal,
            "Issues a new step. Following calls to And are ensured to run after previous ones finished.")
        .def("WaitFor", [](hp::SequenceBuilder& sb, uint64_t v) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.WaitFor(v);
            }, "value"_a, nb::rv_policy::reference_internal,
            "Issues the following steps to wait on the sequence timeline to reach the given value.")
        .def("WaitFor", [](hp::SequenceBuilder& sb, const hp::Timeline& t, uint64_t v) -> hp::SequenceBuilder& {
                nb::gil_scoped_release release;
                return sb.WaitFor(t, v);
            }, "timeline"_a, "value"_a, nb::rv_policy::reference_internal,
            "Issues the following steps to wait for the given timeline to reach the given value")
        .def("printWaitGraph", [](const hp::SequenceBuilder& sb) { return sb.printWaitGraph(); },
            "Returns a visualization of the current wait graph in the form: "
            "(Timeline.ID(WaitValue))* -> (submissions) -> (Timeline.ID(SignalValue)). "
            "Must be called before Submit().")
        .def("Submit", &hp::SequenceBuilder::Submit,
            "Submits the recorded steps as a single batch to the GPU.");
    
    m.def("beginSequence", [](hp::Timeline& t, uint64_t v) { return hp::beginSequence(t, v); },
        "timeline"_a, "startValue"_a = 0, "Starts a new sequence.", nb::rv_policy::move);
    m.def("beginSequence", []() { return hp::beginSequence(getCurrentContext()); },
        "Starts a new sequence.", nb::rv_policy::move);
    
    m.def("execute", [](const hp::Command& cmd) { hp::execute(getCurrentContext(), cmd); },
        nb::call_guard<nb::gil_scoped_release>(),
        "cmd"_a, "Runs the given command synchronous.");
    m.def("execute", [](const hp::Subroutine& sub) { hp::execute(getCurrentContext(), sub); },
        nb::call_guard<nb::gil_scoped_release>(),
        "sub"_a, "Runs the given subroutine synchronous.");
    m.def("executeList", [](nb::list list)
        {
            //minimize time with GIL -> collect commands
            std::vector<const hp::Command*> commands(list.size());
            auto i = 0u;
            for (nb::handle h : list)
                commands[i++] = nb::cast<const hp::Command*>(h);
            
            //release GIL
            nb::gil_scoped_release release;
            //record commands
            hp::execute(getCurrentContext(), [&commands](hp::vulkan::Command& cmd) {
                for (auto c : commands) {
                    c->record(cmd);
                }
            });
        }, "list"_a, "Runs the given of list commands synchronous");
}
