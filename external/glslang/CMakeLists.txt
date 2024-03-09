set(GLSLANG_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/glslang/CInterface/glslang_c_interface.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/GenericCodeGen/CodeGen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/GenericCodeGen/Link.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/preprocessor/Pp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/preprocessor/PpAtom.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/preprocessor/PpContext.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/preprocessor/PpScanner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/preprocessor/PpTokens.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/attribute.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/Constant.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/glslang_tab.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/InfoSink.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/Initialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/Intermediate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/intermOut.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/IntermTraverse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/iomapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/limits.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/linkValidate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/parseConst.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/ParseContextBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/ParseHelper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/PoolAlloc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/propagateNoContraction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/reflection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/RemoveTree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/Scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/ShaderLang.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/SpirvIntrinsics.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/SymbolTable.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/MachineIndependent/Versions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/ResourceLimits/resource_limits_c.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glslang/ResourceLimits/ResourceLimits.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/CInterface/spirv_c_interface.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/disassemble.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/doc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/GlslangToSpv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/InReadableOrder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/Logger.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/SpvBuilder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/SpvPostProcess.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/SPVRemapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SPIRV/SpvTools.cpp
)

if(WIN32)
    list(APPEND GLSLANG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/glslang/OSDependent/Windows/ossource.cpp)
else()
    list(APPEND GLSLANG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/glslang/OSDependent/Unix/ossource.cpp)
endif()

# create always a static library
# since we're using the c++ API, there is no fixed ABI anyway
add_library(glslang STATIC)
target_sources(glslang PRIVATE ${GLSLANG_SOURCES})
target_include_directories(glslang
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)
target_compile_features(glslang PUBLIC cxx_std_17)
set_target_properties(glslang PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(glslang PROPERTIES FOLDER "dependencies")
