#include <nanobind/nanobind.h>

namespace nb = nanobind;

//declare register functions
void registerBufferModule(nb::module_&);
void registerCommandModule(nb::module_&);
void registerCompilerModule(nb::module_&);
void registerContextModule(nb::module_&);
void registerImageModule(nb::module_&);
void registerProgramModule(nb::module_&);
void registerRaytracing(nb::module_&);
void registerStopWatchModule(nb::module_&);
void registerAtomicModule(nb::module_&);
void registerTypeModule(nb::module_&);
void registerDebugModule(nb::module_&);

NB_MODULE(pyhephaistos, m) {
    registerContextModule(m);
    registerCommandModule(m);
    registerCompilerModule(m);
    registerProgramModule(m);
    registerBufferModule(m);
    registerImageModule(m);
    registerStopWatchModule(m);
    registerRaytracing(m);
    registerAtomicModule(m);
    registerTypeModule(m);
    registerDebugModule(m);

#ifdef WARN_LEAKS
    nb::set_leak_warnings(false);
#endif
}
