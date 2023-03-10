#include <nanobind/nanobind.h>

namespace nb = nanobind;

//declare register functions
void registerContextModule(nb::module_&);
void registerBufferModule(nb::module_&);

NB_MODULE(pyhephaistos, m) {
    registerContextModule(m);
    registerBufferModule(m);
}
