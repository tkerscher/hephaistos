#pragma once

#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

//forward
namespace vulkan {
    struct Timeline;
}


class HEPHAISTOS_API Timeline : public Resource {
public:
    [[nodiscard]] uint64_t getValue() const;
    void setValue(uint64_t value);
    void waitValue(uint64_t value) const;
    [[nodiscard]] bool waitValue(uint64_t value, uint64_t timeout) const;

    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    Timeline(Timeline&& other) noexcept;
    Timeline& operator=(Timeline&& other) noexcept;

    Timeline(ContextHandle context, uint64_t initialValue = 0);
    virtual ~Timeline();

public: //internal
    vulkan::Timeline& getTimeline() const;

private:
    std::unique_ptr<vulkan::Timeline> timeline;
};

class HEPHAISTOS_API SequenceBuilder final {
public:
    SequenceBuilder& And(const CommandHandle& command);
    SequenceBuilder& Then(const CommandHandle& command);
    SequenceBuilder& WaitFor(uint64_t value);
    uint64_t Submit();

    SequenceBuilder(Timeline& timeline, uint64_t startValue = 0);
    ~SequenceBuilder();

private:
    struct pImp;
    std::unique_ptr<pImp> _pImp;
};

[[nodiscard]]
inline SequenceBuilder beginSequence(Timeline& timeline, uint64_t startValue = 0) {
    return SequenceBuilder(timeline, startValue);
}

}
