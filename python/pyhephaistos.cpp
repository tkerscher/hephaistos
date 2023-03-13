#include <nanobind/nanobind.h>

namespace nb = nanobind;

//declare register functions
void registerBufferModule(nb::module_&);
void registerCommandModule(nb::module_&);
void registerContextModule(nb::module_&);
void registerProgramModule(nb::module_&);

NB_MODULE(pyhephaistos, m) {
    registerContextModule(m);
    registerCommandModule(m);
    registerProgramModule(m);
    registerBufferModule(m);
}
