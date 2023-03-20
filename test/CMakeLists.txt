# to shorten things
set(TESTROOT ${PROJECT_SOURCE_DIR}/test)

#test files
set(TEST_FILES
    ${TESTROOT}/buffer.cpp
    ${TESTROOT}/command.cpp
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

#create test executable
add_executable(tests ${TEST_FILES})
target_link_libraries(tests PRIVATE hephaistos)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# skip install. There's no point to it
