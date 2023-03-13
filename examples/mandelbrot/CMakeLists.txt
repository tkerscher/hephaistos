add_executable(mandelbrot mandelbrot.cpp mandelbrot.h)
target_link_libraries(mandelbrot PRIVATE hephaistos)
set_target_properties(mandelbrot PROPERTIES FOLDER "examples")

install(
    TARGETS mandelbrot
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT examples
)
