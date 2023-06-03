#pragma once

//fwd
struct VkWriteDescriptorSet;

namespace hephaistos {

class Argument {
public:
    virtual void bindParameter(VkWriteDescriptorSet& binding) const = 0;

    virtual ~Argument() = default;
};

}
