#include "hephaistos/bindings.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>

#include "volk.h"

namespace hephaistos {

namespace {

bool isDescriptorSetEmpty(const VkWriteDescriptorSet& set) {
    return set.pNext == nullptr &&
        set.pImageInfo == nullptr &&
        set.pBufferInfo == nullptr &&
        set.pTexelBufferView == nullptr;
}

}

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

    return !isDescriptorSetEmpty(boundParams[i]);
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
    return !isDescriptorSetEmpty(boundParams[i]);
}

bool BindingTarget::allBindingsBound() const noexcept {
    for (const auto& param : boundParams) {
        if (isDescriptorSetEmpty(param))
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

void BindingTarget::checkAllBindingsBound() const {
    if (std::any_of(
        boundParams.begin(),
        boundParams.end(),
        isDescriptorSetEmpty)
    ) {
        std::ostringstream stream;
        stream << "Cannot dispatch program due to unbound bindings:";
        //collect all unbound bindings to make the error more usefull
        for (auto i = 0u; i < boundParams.size(); ++i) {
            if (!isDescriptorSetEmpty(boundParams[i]))
                continue;

            if (bindingTraits[i].name.empty())
                stream << " " << i;
            else
                stream << " " << bindingTraits[i].name;
        }

        throw std::logic_error(stream.str());
    }
}

}
