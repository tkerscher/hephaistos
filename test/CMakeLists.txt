# to shorten things
set(TESTROOT ${PROJECT_SOURCE_DIR}/test)

#test files
set(TEST_FILES
    ${TESTROOT}/atomic.cpp
    ${TESTROOT}/buffer.cpp
    ${TESTROOT}/command.cpp
    ${TESTROOT}/compiler.cpp
    ${TESTROOT}/context.cpp
    ${TESTROOT}/image.cpp
    ${TESTROOT}/program.cpp
    ${TESTROOT}/raytracing.cpp
)

#fetch catch2 test framework
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.3.2
    GIT_SHALLOW 1
)
FetchContent_MakeAvailable(Catch2)
set_target_properties(Catch2 PROPERTIES FOLDER "dependencies")
set_target_properties(Catch2WithMain PROPERTIES FOLDER "dependencies")

#create test executable
add_executable(tests ${TEST_FILES} ${TESTROOT}/main.cpp)
target_link_libraries(tests PRIVATE hephaistos)
target_link_libraries(tests PRIVATE Catch2::Catch2)

# enable managed resources
target_compile_definitions(hephaistos PUBLIC HEPHAISTOS_MANAGED_RESOURCES)

# skip install. There's no point to it
