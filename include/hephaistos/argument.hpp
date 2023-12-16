#pragma once

//fwd
struct VkWriteDescriptorSet;

namespace hephaistos {

/**
 * @brief Base class for parameters that can be bound to program's bindings
*/
class Argument {
public:
    /**
     * @brief Binds this Argument to the given binding
    */
    virtual void bindParameter(VkWriteDescriptorSet& binding) const = 0;

    virtual ~Argument() = default;
};

}
