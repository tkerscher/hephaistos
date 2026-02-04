#include "hephaistos/compiler.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string.h>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

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

}
