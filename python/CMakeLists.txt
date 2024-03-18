#see nanobind documentation for details

#Collect source files
set(PYROOT ${PROJECT_SOURCE_DIR}/python)
set(SRC
    ${PYROOT}/atomic.cpp
    ${PYROOT}/buffer.cpp
    ${PYROOT}/command.cpp
    ${PYROOT}/compiler.cpp
    ${PYROOT}/context.cpp
    ${PYROOT}/debug.cpp
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

# Create CMake targets for all Python components needed by nanobind
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
else()
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# Run `nanobind.cmake_dir()` from Python to detect where nanobind is installed
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

# Now, import nanobind through CMake's find_package mechanism
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
target_include_directories(pyhephaistos PRIVATE ${PYROOT})
if(WARN_LEAKS)
    target_compile_definitions(pyhephaistos PRIVATE WARN_LEAKS)
endif()

# install
install(TARGETS pyhephaistos LIBRARY DESTINATION hephaistos)
