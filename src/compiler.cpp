#include "hephaistos/compiler.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <spirv_reflect.h>

namespace hephaistos {

namespace {

//default empty header map
const HeaderMap EmptyHeaderMap{};

class Includer : public glslang::TShader::Includer {
public:
    IncludeResult* includeSystem(
        const char* headerName,
        const char* includerName,
        size_t inclusionDepth
    ) override {
        //check all directories
        for (auto& dir : includeDirs) {
            auto path = dir / headerName;
            if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
                //read file into string
                size_t size = std::filesystem::file_size(path);
                std::ifstream file(path, std::ios::binary);
                auto data = new char[size];
                auto name = new std::string(headerName);
                if (file.read(data, size)) {
                    return new IncludeResult(*name, data, size, name);
                }
                else {
                    //error -> clean up
                    delete name;
                    delete[] data;
                    //actually we should return an error saying we failed to load the file
                    //as nullptr means "not found", but it'll do for now
                    return nullptr;
                }
            }
        }

        //nothing found
        return nullptr;
    }

    IncludeResult* includeLocal(
        const char* headerName,
        const char* includerName,
        size_t inclusionDepth
    ) override {
        auto pHeader = headers.find(headerName);
        if (pHeader != headers.end()) {
            return new IncludeResult(
                pHeader->first,
                pHeader->second.c_str(),
                pHeader->second.size(),
                nullptr
            );
        }
        else {
            return nullptr;
        }
    }

    void releaseInclude(IncludeResult* result) override {
        if (!result) return;

        if (result->userData) {
            //result stems from includeSystem
            delete static_cast<std::string*>(result->userData);
            delete[] result->headerData;
        }

        delete result;
    }

    Includer(const std::vector<std::filesystem::path>& includeDirs, const HeaderMap& headers)
        : includeDirs(includeDirs)
        , headers(headers)
    {}

private:
    const std::vector<std::filesystem::path>& includeDirs;
    const HeaderMap& headers;
};

void compileError(const char* reason, glslang::TShader& shader) {
    std::stringstream sstream;
    sstream << reason << '\n' << shader.getInfoLog() << '\n' << shader.getInfoDebugLog();
    throw std::runtime_error(sstream.str());
}

void compileError(const char* reason, glslang::TProgram& program) {
    std::stringstream sstream;
    sstream << reason << '\n' << program.getInfoLog() << '\n' << program.getInfoDebugLog();
    throw std::runtime_error(sstream.str());
}

std::vector<uint32_t> compileImpl(
    std::string_view code,
    ShaderStage stage,
    const std::vector<std::filesystem::path>& includeDirs,
    const HeaderMap& headers = EmptyHeaderMap
) {
    auto language = static_cast<EShLanguage>(stage);
    glslang::TShader shader(language);
    auto cstr_code = code.data();
    shader.setStrings(&cstr_code, 1);

    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    shader.setAutoMapBindings(true);

    std::string preprocessedGLSL;
    Includer includer(includeDirs, headers);
    auto success = shader.preprocess(
        GetDefaultResources(),
        460, ENoProfile,
        false, false,
        EShMsgDefault,
        &preprocessedGLSL,
        includer);
    if (!success) compileError("GLSL preprocessing failed!", shader);
    auto preprocessedCStr = preprocessedGLSL.c_str();
    shader.setStrings(&preprocessedCStr, 1);

    if (!shader.parse(GetDefaultResources(), 460, false, EShMsgDefault))
        compileError("GLSL parsing failed!", shader);

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(EShMsgDefault))
        compileError("GLSL linking failed!", program);

    if (!program.mapIO())
        compileError("GLSL binding remapping failed!", program);

    glslang::SpvOptions spvOptions{
        .optimizeSize = true,
        .validate = true
    };
    std::vector<uint32_t> spirv;
    const auto intermediate = program.getIntermediate(language);
    glslang::GlslangToSpv(*intermediate, spirv, &spvOptions);

    return spirv;
}

//TODO: duplicate code from vk/reflection.cpp -> extract into common file

#define STR(c) case SPV_REFLECT_RESULT_##c: return "Reflect: "#c;

inline auto printSpvResult(SpvReflectResult result) {
    switch (result) {
        STR(SUCCESS)
        STR(NOT_READY)
        STR(ERROR_PARSE_FAILED)
        STR(ERROR_ALLOC_FAILED)
        STR(ERROR_RANGE_EXCEEDED)
        STR(ERROR_NULL_POINTER)
        STR(ERROR_INTERNAL_ERROR)
        STR(ERROR_COUNT_MISMATCH)
        STR(ERROR_ELEMENT_NOT_FOUND)
        STR(ERROR_SPIRV_INVALID_CODE_SIZE)
        STR(ERROR_SPIRV_INVALID_MAGIC_NUMBER)
        STR(ERROR_SPIRV_UNEXPECTED_EOF)
        STR(ERROR_SPIRV_INVALID_ID_REFERENCE)
        STR(ERROR_SPIRV_SET_NUMBER_OVERFLOW)
        STR(ERROR_SPIRV_INVALID_STORAGE_CLASS)
        STR(ERROR_SPIRV_RECURSION)
        STR(ERROR_SPIRV_INVALID_INSTRUCTION)
        STR(ERROR_SPIRV_UNEXPECTED_BLOCK_DATA)
        STR(ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE)
        STR(ERROR_SPIRV_INVALID_ENTRY_POINT)
        STR(ERROR_SPIRV_INVALID_EXECUTION_MODE)
    default:
        return "Unknown result code";
    }
}

#undef STR

void spvCheckResult(SpvReflectResult result) {
    if (result != SPV_REFLECT_RESULT_SUCCESS)
        throw std::runtime_error(printSpvResult(result));
}

}

std::vector<uint32_t> Compiler::compile(std::string_view code, ShaderStage stage) const {
    return compileImpl(code, stage, includeDirs);
}
std::vector<uint32_t> Compiler::compile(std::string_view code, const HeaderMap& headers, ShaderStage stage) const {
    return compileImpl(code, stage, includeDirs, headers);
}

void Compiler::addIncludeDir(std::filesystem::path path) {
    includeDirs.emplace_back(std::move(path));
}
void Compiler::popIncludeDir() {
    includeDirs.pop_back();
}
void Compiler::clearIncludeDir() {
    includeDirs.clear();
}

Compiler& Compiler::operator=(Compiler&&) noexcept = default;
Compiler::Compiler(Compiler&&) noexcept = default;

Compiler::Compiler() {
    glslang::InitializeProcess();
}

Compiler::~Compiler() {
    glslang::FinalizeProcess();
}

struct CompilerSession::Imp {
    const Compiler& compiler;
    std::unordered_map<std::string, uint32_t> bindings;
    std::unordered_set<uint32_t> slots;

    void process(std::vector<uint32_t>& code);

    Imp(const Compiler& compiler)
        : compiler(compiler)
        , bindings()
    {}
};

void CompilerSession::Imp::process(std::vector<uint32_t>& code) {
    //reflect code
    SpvReflectShaderModule reflectModule;
    spvCheckResult(spvReflectCreateShaderModule2(
        SPV_REFLECT_MODULE_FLAG_NO_COPY,
        4 * code.size(),
        code.data(),
        &reflectModule
    ));

    //we only support a single descriptor set
    if (reflectModule.descriptor_set_count > 1)
        throw std::runtime_error("Only a single descriptor set is supported!");
    if (reflectModule.descriptor_set_count == 0) {
        //nothing to fix
        spvReflectDestroyShaderModule(&reflectModule);
        return;
    }

    //this will later contain bindings we have to update
    std::unordered_map<uint32_t, uint32_t> updates;

    //update common program layout
    auto& set = reflectModule.descriptor_sets[0];
    auto pBinding = set.bindings[0];
    auto pBindingEnd = pBinding + set.binding_count;
    for (; pBinding != pBindingEnd; ++pBinding) {
        //skip unused bindings. auto mapping assign binding zero to these.
        if (!pBinding->accessed) continue;

        if (pBinding->count == 0)
            throw std::runtime_error("Unbound arrays are not supported!");

        //fetch binding name
        auto binding = pBinding->binding;
        auto name = pBinding->name ? pBinding->name : nullptr;
        constexpr auto blockType =
            SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER |
            SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if ((pBinding->descriptor_type & blockType) != 0) {
            if (pBinding->type_description->type_name)
                name = pBinding->type_description->type_name;
            else
                name = nullptr;
        }
        //we cant ensure a common pipeline layout without names
        if (!name)
            throw std::runtime_error(std::format("Binding {} has no name!", binding));

        //check if encountered that name already
        const auto pMap = bindings.find(name);
        if (pMap != bindings.end()) {
            //and if so, if it matches the expected binding
            if (pMap->second != binding) {
                //at the corresponding binding to the todo list
                updates.emplace(binding, pMap->second);
            }
        }
        else {
            //new name. check if the assigned binding is actually free
            if (slots.contains(binding)) {
                //find next free slot
                uint32_t newBinding = 0u;
                while (slots.contains(newBinding)) ++newBinding;                
                //mark to update
                updates.emplace(binding, newBinding);
                binding = newBinding;
            }
            //a new name -> write it down for later retrieval
            bindings.emplace(name, binding);
            slots.insert(binding);
        }
    }

    //by now we are finished with reflecting
    spvReflectDestroyShaderModule(&reflectModule);

    //anything to do?
    if (updates.empty()) return;

    //SPIR-V is surprisingly easy to traverse
    //to make our lifes easier, we simply rewrite all OpDecorate Bindings
    //with a binding number we want to replace
    auto pCode = code.data();
    auto pCodeEnd = pCode + code.size();
    pCode += 5; //first 20 bytes (5 words) are header
    while (pCode < pCodeEnd) {
        //decode instruction
        uint32_t length = *pCode >> 16;
        uint32_t opCode = *pCode & 0xFFFFu;
        //check for OpDecorate and DecorateBinding
        if (opCode == 0x47u && *(pCode + 2) == 0x21u) {
            auto oldBinding = *(pCode + 3);
            auto pUpdate = updates.find(oldBinding);
            if (pUpdate != updates.end())
                *(pCode + 3) = pUpdate->second;
        }
        //go to next op
        pCode += length;
    }
}

std::vector<uint32_t> CompilerSession::compile(
    std::string_view code,
    ShaderStage stage
) {
    return compile(code, EmptyHeaderMap, stage);
}

std::vector<uint32_t> CompilerSession::compile(
    std::string_view source,
    const HeaderMap& headers,
    ShaderStage stage
) {
    auto code = compileImpl(source, stage, pImp->compiler.includeDirs, headers);
    pImp->process(code);
    return code;
}

CompilerSession::CompilerSession(CompilerSession&&) noexcept = default;
CompilerSession& CompilerSession::operator=(CompilerSession&&) noexcept = default;

CompilerSession::CompilerSession(const Compiler& compiler)
    : pImp(std::make_unique<Imp>(compiler))
{}
CompilerSession::~CompilerSession() = default;

}
