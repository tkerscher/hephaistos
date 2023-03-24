#pragma once

#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

//forward
namespace vulkan {
    struct Command;
    struct Timeline;
}
struct SubmissionResources;

class HEPHAISTOS_API Command {
public:
    virtual void record(vulkan::Command& cmd) const = 0;
    virtual ~Command();
};

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

    explicit Timeline(ContextHandle context, uint64_t initialValue = 0);
    virtual ~Timeline();

public: //internal
    vulkan::Timeline& getTimeline() const;

private:
    std::unique_ptr<vulkan::Timeline> timeline;
};

class HEPHAISTOS_API Submission {
public:
    const Timeline& getTimeline() const;
    uint64_t getFinalStep() const;

    void wait() const;
    [[nodiscard]] bool wait(uint64_t timeout) const;

    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;

    Submission(Submission&& other) noexcept;
    Submission& operator=(Submission&& other) noexcept;

    Submission(const Timeline& timeline, uint64_t finalStep,
        std::unique_ptr<SubmissionResources> resources);
    ~Submission();

private:
    uint64_t finalStep;
    std::reference_wrapper<const Timeline> timeline;
    std::unique_ptr<SubmissionResources> resources;
};

class HEPHAISTOS_API SequenceBuilder final {
public:
    SequenceBuilder& And(const Command& command);
    SequenceBuilder& Then(const Command& command);
    SequenceBuilder& WaitFor(uint64_t value);
    Submission Submit();

    SequenceBuilder(SequenceBuilder&) = delete;
    SequenceBuilder& operator=(SequenceBuilder&) = delete;

    SequenceBuilder(SequenceBuilder&& other) noexcept;
    SequenceBuilder& operator=(SequenceBuilder&& other) noexcept;

    explicit SequenceBuilder(ContextHandle context);
    explicit SequenceBuilder(Timeline& timeline, uint64_t startValue = 0);
    ~SequenceBuilder();

private:
    struct pImp;
    std::unique_ptr<pImp> _pImp;
};

[[nodiscard]]
inline SequenceBuilder beginSequence(Timeline& timeline, uint64_t startValue = 0) {
    return SequenceBuilder(timeline, startValue);
}
[[nodiscard]]
inline SequenceBuilder beginSequence(const ContextHandle& context) {
    return SequenceBuilder(context);
}

}
