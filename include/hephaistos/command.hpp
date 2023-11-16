#pragma once

#include <concepts>
#include <functional>
#include <string>
#include <utility>

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

class HEPHAISTOS_API Subroutine : public Resource {
public:
    bool simultaneousUse() const;
    const vulkan::Command& getCommandBuffer() const;

    Subroutine(const Subroutine&) = delete;
    Subroutine& operator=(const Subroutine&) = delete;

    Subroutine(Subroutine&& other) noexcept;
    Subroutine& operator=(Subroutine&& other) noexcept;

    ~Subroutine() override;

private:
    Subroutine(
        ContextHandle context,
        std::unique_ptr<vulkan::Command> cmdBuffer,
        bool simultaneous_use);

    friend class SubroutineBuilder;

private:
    std::unique_ptr<vulkan::Command> cmdBuffer;
    bool simultaneous_use;
};

class HEPHAISTOS_API SubroutineBuilder final {
public:
    explicit operator bool() const;

    SubroutineBuilder& addCommand(const Command& command) &;
    SubroutineBuilder addCommand(const Command& command) &&;
    Subroutine finish();

    SubroutineBuilder(const SubroutineBuilder& other) = delete;
    SubroutineBuilder& operator=(const SubroutineBuilder& other) = delete;

    SubroutineBuilder(SubroutineBuilder&& other) noexcept;
    SubroutineBuilder& operator=(SubroutineBuilder&& other) noexcept;

    explicit SubroutineBuilder(ContextHandle context, bool simultaneous_use = false);
    ~SubroutineBuilder();

private:
    ContextHandle context;
    std::unique_ptr<vulkan::Command> cmdBuffer;
    bool simultaneous_use;
};

struct simultaneous_use_tag{};
inline constexpr simultaneous_use_tag simultaneous_use{};

template<std::derived_from<Command> ...T>
[[nodiscard]] Subroutine createSubroutine(ContextHandle context, T... commands) {
    SubroutineBuilder builder(std::move(context));
    (builder.addCommand(commands), ...);
    return builder.finish();
}
template<std::derived_from<Command> ...T>
[[nodiscard]] Subroutine createSubroutine(ContextHandle context, simultaneous_use_tag, T... commands) {
    SubroutineBuilder builder(std::move(context), true);
    (builder.addCommand(commands), ...);
    return builder.finish();
}

class HEPHAISTOS_API Timeline : public Resource {
public:
    [[nodiscard]] uint64_t getId() const;

    [[nodiscard]] uint64_t getValue() const;
    void setValue(uint64_t value);
    void waitValue(uint64_t value) const;
    [[nodiscard]] bool waitValue(uint64_t value, uint64_t timeout) const;

    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    Timeline(Timeline&& other) noexcept;
    Timeline& operator=(Timeline&& other) noexcept;

    explicit Timeline(ContextHandle context, uint64_t initialValue = 0);
    ~Timeline() override;

public: //internal
    vulkan::Timeline& getTimeline() const;

private:
    std::unique_ptr<vulkan::Timeline> timeline;
};

class HEPHAISTOS_API Submission {
public:
    const Timeline& getTimeline() const;
    uint64_t getFinalStep() const;

    //true, if submission manages no internal resources
    //i.e. destruction causes no wait on completion
    [[nodiscard]] bool forgettable() const noexcept;

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
    explicit operator bool() const;

    SequenceBuilder& And(const Command& command) &;
    SequenceBuilder& And(const Subroutine& subroutine) &;
    template<class ...T>
    SequenceBuilder& AndList(const T& ...steps) & {
        (And(steps), ...);
        return *this;
    }
    SequenceBuilder& NextStep() &;
    SequenceBuilder& Then(const Command& command) &;
    SequenceBuilder& Then(const Subroutine& subroutine) &;
    SequenceBuilder& WaitFor(uint64_t value) &;
    SequenceBuilder& WaitFor(const Timeline& timeline, uint64_t value) &;

    SequenceBuilder And(const Command& command) &&;
    SequenceBuilder And(const Subroutine& subroutine) &&;
    template<class ...T>
    SequenceBuilder AndList(const T& ...steps) && {
        (And(steps), ...);
        return std::move(*this);
    }
    SequenceBuilder NextStep() &&;
    SequenceBuilder Then(const Command& command) &&;
    SequenceBuilder Then(const Subroutine& subroutine) &&;
    SequenceBuilder WaitFor(uint64_t value) &&;
    SequenceBuilder WaitFor(const Timeline& timeline, uint64_t value) &&;

    Submission Submit();

    std::string printWaitGraph() const;

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

HEPHAISTOS_API void execute(const ContextHandle& context, const Command& command);
HEPHAISTOS_API void execute(const ContextHandle& context, const Subroutine& subroutine);
HEPHAISTOS_API void execute(const ContextHandle& context,
    const std::function<void(vulkan::Command& cmd)>& emitter);
template<std::derived_from<Command> ...T>
void executeList(const ContextHandle& context, const T& ...commands) {
    execute(context, [&commands...](vulkan::Command& cmd) {
        (commands.record(cmd), ...);
    });
}

}
