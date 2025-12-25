#pragma once

#include <optional>
#include <string>
#include <vector>

#include "hephaistos/config.hpp"
#include "hephaistos/imageformat.hpp"

//Forward from Vulkan API
struct VkWriteDescriptorSet;

namespace hephaistos {

/**
 * @brief Parameter Type
 * 
 * Enumeration of parameter types a binding inside a program can define.
 * Used for reflections.
*/
enum class ParameterType {
    //SAMPLER = 0,
    
    COMBINED_IMAGE_SAMPLER = 1,
    //SAMPLED_IMAGE = 2,
    
    STORAGE_IMAGE = 3,
    //UNIFORM_TEXEL_BUFFER = 4,
    //STORAGE_TEXEL_BUFFER = 5,
    
    UNIFORM_BUFFER = 6,
    STORAGE_BUFFER = 7,
    //UNIFORM_BUFFER_DYNAMIC = 8,
    //STORAGE_BUFFER_DYNAMIC = 9,
    
    ACCELERATION_STRUCTURE = 1000150000
};

/**
 * @brief Image properties a binding defines
*/
struct ImageBindingTraits {
    /**
     * @brief Format of the image binding
    */
    ImageFormat format;
    /**
     * @brief Number of dimension the image has
    */
    uint8_t dims;

    bool operator==(const ImageBindingTraits&) const = default;
    bool operator!=(const ImageBindingTraits&) const = default;
};

/**
 * @brief Properties of a binding inside a program
*/
struct BindingTraits {
    /**
     * @brief Name of the binding
     * 
     * Name of the binding as defined in the shader code.
     * Can be used to bind parameter to this binding in a program.
     * The compiler may have stripped away this information, where this becomes
     * an empty string.
    */
    std::string name;
    /**
     * @brief Binding number
     * 
     * Binding number, i.e. its location, as defined in the shader.
     * Can be used to bind parameter to this binding in a program and will
     * always be present.
    */
    uint32_t binding;
    /**
     * @brief Type of the binding
    */
    ParameterType type;
    /**
     * @brief Image properties of the binding
     * 
     * Only set if the binding type expects an image or texture
    */
    std::optional<ImageBindingTraits> imageTraits;
    /**
     * @brief Multiplicity of the binding
     * 
     * For plain bindings this will be one, whereas for array bindings, this
     * will give the size of the array. A value of zero denotes a runtime array.
    */
    uint32_t count;

    bool operator==(const BindingTraits&) const = default;
    bool operator!=(const BindingTraits&) const = default;
};

/**
 * @brief Base class for targets to which parameters can be bound.
 */
class HEPHAISTOS_API BindingTarget {
public:
    /**
     * @brief Returns the amount of bindings present in this program
     * 
     * @return Amount of bindings in this program
    */
    [[nodiscard]] uint32_t getBindingCount() const noexcept;

    /**
     * @brief Tests whether the program has a binding of the given name
     * 
     * @return True, if a binding of given name is present
    */
    [[nodiscard]] bool hasBinding(std::string_view name) const noexcept;

    /**
     * @brief Returns the traits of the given binding
     * 
     * Reflects the binding specified by its binding number in the shader code.
     * Throws if the given binding does not exist.
     * 
     * @param i Binding number to be reflected
     * @return BindingTraits of the given binding
    */
    [[nodiscard]] const BindingTraits& getBindingTraits(uint32_t i) const;
    /**
     * @brief Returns the traits of the given binding
     * 
     * Reflects the binding specified by its name as defined in the shader code.
     * Throws if the given binding cannot be found.
     * @note Shader compiler may strip away the binding names
     * 
     * @param name Name of the binding to be reflected
     * @return BindingTraits of the given binding
    */
    [[nodiscard]] const BindingTraits& getBindingTraits(std::string_view name) const;

    /**
     * @brief Checks whether the given binding is currently bound
     * 
     * @param i Binding number to be checked
     * @return True, if the binding is currently bound, false otherwise
     */
    [[nodiscard]] bool isBindingBound(uint32_t i) const;
    /**
     * @brief Checks wether the given binding is currently bound
     * 
     * Checks the binding specified by its name as defined in the shader code.
     * Throws if the given binding cannot be found.
     * @note Shader compiler may strip away the binding names
     * 
     * @param name Name of the binding to be checked
     * @return True, if the binding is currently bound, false otherwise
     */
    [[nodiscard]] bool isBindingBound(std::string_view name) const;
    /**
     * @brief Checks whether all bindings are currently bound
     * 
     * @return True, if all bindings are bound, false otherwise.
    */
    [[nodiscard]] bool allBindingsBound() const noexcept;

    /**
     * @brief Returns a list of BindingTraits of all bindings
     * 
     * @return Vector of BindingTraits one for each binding
    */
    [[nodiscard]] const std::vector<BindingTraits>& listBindings() const noexcept;

    /**
     * @brief Binds the given parameter
     * 
     * @param param Parameter to bind
     * @param binding Binding number to bind the parameter to
    */
    template<class T>
    void bindParameter(const T& param, uint32_t binding) {
        param.bindParameter(getBinding(binding));
    }
    /**
     * @brief Binds the given parameter
     * 
     * @param param Parameter to bind
     * @param binding Name of the binding to bind the parameter to
    */
    template<class T>
    void bindParameter(const T& param, std::string_view name) {
        param.bindParameter(getBinding(name));
    }
    /**
     * @brief Binds the list of parameters
     * 
     * Binds the given list of parameters to bindings in the order as they are
     * defined in the program.
     * 
     * @param param... List of parameters to bind
    */
    template<class ...T>
    void bindParameterList(const T& ...param) {
        uint32_t binding = 0;
        (param.bindParameter(getBinding(binding++)), ...);
    }

protected: //internal 
    [[nodiscard]] VkWriteDescriptorSet& getBinding(uint32_t i);
    [[nodiscard]] VkWriteDescriptorSet& getBinding(std::string_view name);

    /**
     * @brief Checks whether all bindings are bound and throws if not
    */
    void checkAllBindingsBound() const;

protected:
    std::vector<BindingTraits> bindingTraits;
    std::vector<VkWriteDescriptorSet> boundParams;
};

}
