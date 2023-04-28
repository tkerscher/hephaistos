add_executable(sampling sampling.cpp sampling.h)
target_link_libraries(sampling PRIVATE hephaistos)
set_target_properties(sampling PROPERTIES FOLDER "examples")

install(
    TARGETS sampling
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT examples
)
