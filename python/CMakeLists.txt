#see nanobind documentation for details

#Collect source files
set(PYROOT ${PROJECT_SOURCE_DIR}/python)
set(SRC
    ${PYROOT}/atomic.cpp
    ${PYROOT}/bindings.cpp
    ${PYROOT}/buffer.cpp
    ${PYROOT}/command.cpp
    ${PYROOT}/compiler.cpp
    ${PYROOT}/context.cpp
    ${PYROOT}/debug.cpp
    ${PYROOT}/exception.cpp
    ${PYROOT}/image.cpp
    ${PYROOT}/program.cpp
    ${PYROOT}/pyhephaistos.cpp
    ${PYROOT}/raytracing.cpp
    ${PYROOT}/stopwatch.cpp
    ${PYROOT}/types.cpp
)

# Fix missing shared library name for cibuildwheel+windows+pypy3.9
if (MSVC AND NOT PYTHON_LIBRARY AND (${PYTHON_VERSION_STRING} MATCHES "3.9."))
    get_filename_component(PYTHON_LIBRARY ${PYTHON_INCLUDE_DIR} DIRECTORY)
    set(PYTHON_LIBRARY "${PYTHON_LIBRARY}/libs/python39.lib")
endif()

set(Python_VERSION "${PYTHON_VERSION_STRING}")
set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
set(Python_LIBRARY "${PYTHON_LIBRARY}")

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
    OPTIONAL_COMPONENTS Development.SABIModule)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
    pyhephaistos
    NB_DOMAIN hephaistos
    STABLE_ABI
    NB_STATIC
    LTO
    ${SRC}
)  
target_link_libraries(pyhephaistos PRIVATE hephaistos)
target_compile_definitions(hephaistos PUBLIC HEPHAISTOS_MANAGED_RESOURCES)
target_include_directories(pyhephaistos PRIVATE ${PYROOT})
if(WARN_LEAKS)
    target_compile_definitions(pyhephaistos PRIVATE WARN_LEAKS)
endif()

# create stubs
nanobind_add_stub(
    pyhephaistos_stub
    MODULE pyhephaistos
    OUTPUT pyhephaistos.pyi
    MARKER_FILE py.typed
    PYTHON_PATH $<TARGET_FILE_DIR:pyhephaistos>
    DEPENDS pyhephaistos
)

# install
install(TARGETS pyhephaistos LIBRARY DESTINATION hephaistos)
install(FILES
    ${CMAKE_BINARY_DIR}/python/py.typed
    ${CMAKE_BINARY_DIR}/python/pyhephaistos.pyi
    DESTINATION hephaistos
)
