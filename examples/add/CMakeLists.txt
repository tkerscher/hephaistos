add_executable(add add.cpp add.h)
target_link_libraries(add PRIVATE hephaistos)
set_target_properties(add PROPERTIES FOLDER "examples")

install(
    TARGETS add
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT examples
)
