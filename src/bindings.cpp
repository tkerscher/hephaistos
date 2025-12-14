#include "hephaistos/bindings.hpp"

#include <stdexcept>
#include <sstream>

#include "volk.h"

#include "vk/descriptor.hpp"

namespace hephaistos {

uint32_t BindingTarget::getBindingCount() const noexcept {
	return bindingTraits.size();
}

bool BindingTarget::hasBinding(std::string_view name) const noexcept {
    auto it = std::find_if(
        bindingTraits.begin(),
        bindingTraits.end(),
        [name](const BindingTraits& t) -> bool {
            return t.name == name;
        });
    return it == bindingTraits.end();
}

const BindingTraits& BindingTarget::getBindingTraits(uint32_t i) const {
    if (i >= bindingTraits.size())
        throw std::runtime_error("There is no binding point at specified number!");

    return bindingTraits[i];
}

const BindingTraits& BindingTarget::getBindingTraits(std::string_view name) const {
    auto it = std::find_if(
        bindingTraits.begin(),
        bindingTraits.end(),
        [name](const BindingTraits& t) -> bool {
            return t.name == name;
        });

    if (it == bindingTraits.end())
        throw std::runtime_error("There is no binding point at specified location!");

    return *it;
}

bool BindingTarget::isBindingBound(uint32_t i) const {
    if (i >= boundParams.size())
        throw std::runtime_error("There is no binding point at specified number!");

    return !vulkan::isDescriptorSetEmpty(boundParams[i]);
}

bool BindingTarget::isBindingBound(std::string_view name) const {
    auto it = std::find_if(
        bindingTraits.begin(),
        bindingTraits.end(),
        [name](const BindingTraits& t) -> bool {
            return t.name == name;
        });

    if (it == bindingTraits.end())
        throw std::runtime_error("There is no binding point at specified location! Binding name: " + std::string(name));

    auto i = std::distance(bindingTraits.begin(), it);
    return !vulkan::isDescriptorSetEmpty(boundParams[i]);
}

bool BindingTarget::allBindingsBound() const noexcept {
    for (const auto& param : boundParams) {
        if (vulkan::isDescriptorSetEmpty(param))
            return false;
    }

    return true;
}

const std::vector<BindingTraits>& BindingTarget::listBindings() const noexcept {
    return bindingTraits;
}

VkWriteDescriptorSet& BindingTarget::getBinding(uint32_t i) {
    if (i >= boundParams.size())
        throw std::runtime_error("There is no binding point at specified number! Binding: " + std::to_string(i));

    return boundParams[i];
}

VkWriteDescriptorSet& BindingTarget::getBinding(std::string_view name) {
    auto it = std::find_if(
        bindingTraits.begin(),
        bindingTraits.end(),
        [name](const BindingTraits& t) -> bool {
            return t.name == name;
        });

    if (it == bindingTraits.end())
        throw std::runtime_error("There is no binding point at specified location! Binding name: " + std::string(name));

    auto i = std::distance(bindingTraits.begin(), it);
    return boundParams[i];
}

}
