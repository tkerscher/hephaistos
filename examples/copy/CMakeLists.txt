add_executable(copy copy.cpp)
target_link_libraries(copy PRIVATE hephaistos)
set_target_properties(copy PROPERTIES FOLDER "examples")

install(
	TARGETS copy
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	COMPONENT examples
)
