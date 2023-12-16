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

/**
 * @brief Base class for all commands
 * 
 * Commands are used to record work into buffers before submitting them to the
 * device. Work on the device is therefore completely asynchronous.
*/
class HEPHAISTOS_API Command {
public:
    /**
     * @brief Records this command onto the given command buffer
    */
    virtual void record(vulkan::Command& cmd) const = 0;
    virtual ~Command();
};

/**
 * @brief Reusable sequence of commands
 * 
 * Recording work onto command buffers have a non negligible CPU overhead.
 * Subroutines allow to reuse common sequences of commands to amortize this
 * overhead.
*/
class HEPHAISTOS_API Subroutine : public Resource {
public:
    /**
     * @brief Wether this Subroutine can be submitted multiple times simultaneously
    */
    bool simultaneousUse() const;

    Subroutine(const Subroutine&) = delete;
    Subroutine& operator=(const Subroutine&) = delete;

    Subroutine(Subroutine&& other) noexcept;
    Subroutine& operator=(Subroutine&& other) noexcept;

    ~Subroutine() override;

public: //internal
    const vulkan::Command& getCommandBuffer() const;

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

/**
 * @brief Builder for creating Subroutines from a sequence of Command
*/
class HEPHAISTOS_API SubroutineBuilder final {
public:
    /**
     * @brief True, if the builder is still recording
    */
    explicit operator bool() const;

    /**
     * @brief Records the next command into the sequence
    */
    SubroutineBuilder& addCommand(const Command& command) &;
    /**
     * @brief Records the next command into the sequence
    */
    SubroutineBuilder addCommand(const Command& command) &&;
    /**
     * @brief Finishes recording and returns the built Subroutine
    */
    Subroutine finish();

    SubroutineBuilder(const SubroutineBuilder& other) = delete;
    SubroutineBuilder& operator=(const SubroutineBuilder& other) = delete;

    SubroutineBuilder(SubroutineBuilder&& other) noexcept;
    SubroutineBuilder& operator=(SubroutineBuilder&& other) noexcept;

    /**
     * @brief Creates a new SubroutineBuilder
     * 
     * @param context Conext onto which to create the builder
     * @param simultaneous_use Wether the built Subroutine can be submitted
     *                         multiple times simultaneously
    */
    explicit SubroutineBuilder(ContextHandle context, bool simultaneous_use = false);
    ~SubroutineBuilder();

private:
    ContextHandle context;
    std::unique_ptr<vulkan::Command> cmdBuffer;
    bool simultaneous_use;
};

/**
 * @brief Tag to enable simultaneous use
*/
struct simultaneous_use_tag{};
/**
 * @brief Tag to enable simultaneous use
*/
inline constexpr simultaneous_use_tag simultaneous_use{};

/**
 * @brief Creates a new subroutine from the given sequence of commands
 * 
 * @param context Context onto which to create the Subroutine
 * @param commands... Sequence of Command to record
 * @return Subroutine consisting of the provided sequence of Command
*/
template<std::derived_from<Command> ...T>
[[nodiscard]] Subroutine createSubroutine(ContextHandle context, T... commands) {
    SubroutineBuilder builder(std::move(context));
    (builder.addCommand(commands), ...);
    return builder.finish();
}
/**
 * @brief Creates a new subroutine from the given sequence of commands and sets
 *        the simultaneous use flag
 * 
 * @param context Context onto which to create the Subroutine
 * @param commands... Sequence of Command to record
 * @return Subroutine consisting of the provided sequence of Command
*/
template<std::derived_from<Command> ...T>
[[nodiscard]] Subroutine createSubroutine(ContextHandle context, simultaneous_use_tag, T... commands) {
    SubroutineBuilder builder(std::move(context), true);
    (builder.addCommand(commands), ...);
    return builder.finish();
}

/**
 * @brief Synchronizes work between and across GPU and CPU
 * 
 * Timeline allows to synchronize work between GPU-GPU, CPU-CPU and GPU-CPU.
 * It does that via an internal counter, whose value can be queried to get the
 * current progress without the need of extra synchronization. The counter can
 * be incremented both on CPU and GPU.
*/
class HEPHAISTOS_API Timeline : public Resource {
public:
    /**
     * @brief Id of this timeline
    */
    [[nodiscard]] uint64_t getId() const;

    /**
     * @brief Queries the current value of the timeline
    */
    [[nodiscard]] uint64_t getValue() const;
    /**
     * @brief Sets the value of the timeline
     * 
     * @note Decreasing the current value results in undefined behaviour
     *
     * @param value Value to set the timeline to
    */
    void setValue(uint64_t value);
    /**
     * @brief Blocks the calling code until the timeline reaches the given value
     * 
     * @param value Value to wait the timeline to reach
    */
    void waitValue(uint64_t value) const;
    /**
     * @brief Blocks the calling code until the timeline reaches the given value
     *        or the timeout expires
     * 
     * @param value Value to wait the timeline to reach
     * @param timeout Timeout in nanoseconds to wait for
     * @return True, if the value was reached, false if timeout
    */
    [[nodiscard]] bool waitValue(uint64_t value, uint64_t timeout) const;

    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    Timeline(Timeline&& other) noexcept;
    Timeline& operator=(Timeline&& other) noexcept;

    /**
     * @brief Creates a new Timeline
     * 
     * @param context Context onto which to create the Timeline
     * @param initialValue Initial value of the timeline
    */
    explicit Timeline(ContextHandle context, uint64_t initialValue = 0);
    ~Timeline() override;

public: //internal
    vulkan::Timeline& getTimeline() const;

private:
    std::unique_ptr<vulkan::Timeline> timeline;
};

/**
 * @brief Submission allow to wait for submitted work to finish
*/
class HEPHAISTOS_API Submission {
public:
    /**
     * @brief Returns the Timeline used to synchronize the submitted work
    */
    const Timeline& getTimeline() const;
    /**
     * @brief Returns the value the Timeline reaches once the work finishes
    */
    uint64_t getFinalStep() const;

    /**
     * @brief If true, Submission can be destructed without any wait
     * 
     * Submission may manage internal resources, which can only be destroyed
     * once the submitted work has finished. If true, the Submission does not
     * manage any such resources and can be destructed immediately, otherwise
     * the destructor waits on the work to finish.
    */
    [[nodiscard]] bool forgettable() const noexcept;

    /**
     * @brief Blocks calling code until submitted work has finished
    */
    void wait() const;
    /**
     * @brief Blocks calling code until submitted work has finished or the
     *        timeout expires
     * 
     * @param timeout Timeout in nanoseconds to wait for the work to finish
     * @return True, if the work has finished, false if the timeout expires
    */
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

/**
 * @brief Builder for creating work to be submitted to the device
*/
class HEPHAISTOS_API SequenceBuilder final {
public:
    /**
     * @brief True, if the builder is still recording
    */
    explicit operator bool() const;

    /**
     * @brief Records the given command to run in the current step
     * 
     * Recording command in the same step allows them to run in parallel if no
     * further synchronization creates a dependency between them
     * 
     * @param command Command to record
    */
    SequenceBuilder& And(const Command& command) &;
    /**
     * @brief Records the given subroutine to run in the current step
     * 
     * Recording a subroutine in the same step allows it to run in parallel
     * with other commands and subroutines.
     * 
     * @param subroutine Subroutine to record
    */
    SequenceBuilder& And(const Subroutine& subroutine) &;
    template<class ...T>
    /**
     * @brief Records the given list of Command and Subroutine in the current
     *        step
     * 
     * Recording in the same step allows to run work in parallel.
     * 
     * @param steps... Command and Subroutine to record in the current step
    */
    SequenceBuilder& AndList(const T& ...steps) & {
        (And(steps), ...);
        return *this;
    }
    /**
     * @brief Finalizes the current step and prepares the next one
     * 
     * Individual steps are guaranteed to run in sequence, i.e. previous steps
     * are finished before the next one is started.
    */
    SequenceBuilder& NextStep() &;
    /**
     * @brief Finalizes the current step and records the given Command in the
     *        next one
     * 
     * @param command Command to record
    */
    SequenceBuilder& Then(const Command& command) &;
    /**
     * @brief Finalizes the current step and records the given Subroutine in the
     *        next one
     * 
     * @param subroutine Subroutine to be recorded
    */
    SequenceBuilder& Then(const Subroutine& subroutine) &;
    /**
     * @brief Finalizes the current step and issues the next step to not start
     *        until the underlying Timeline reaches at least the given value
     * 
     * @param value Value to wait for
    */
    SequenceBuilder& WaitFor(uint64_t value) &;
    /**
     * @brief Finalizes the current step and issues the next step to not start
     *        until the given Timeline reaches at least the given value
     * 
     * @note Multiple waits can be combined for the same step but must happen
     *       before any work has been recorded in the current step
     * 
     * @param timeline Timeline to wait on
     * @param value Value to wait for
    */
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

    /**
     * @brief Submits the recorded work to the device
     * 
     * @return Submission allowing to wait on the work to finish
    */
    Submission Submit();

    /**
     * @brief Creates a human readable representation of recorded steps
    */
    std::string printWaitGraph() const;

    SequenceBuilder(SequenceBuilder&) = delete;
    SequenceBuilder& operator=(SequenceBuilder&) = delete;

    SequenceBuilder(SequenceBuilder&& other) noexcept;
    SequenceBuilder& operator=(SequenceBuilder&& other) noexcept;

    /**
     * @brief Creates a new SequenceBuilder
     * 
     * @note Creates an internal Timeline used for synchronizing steps
     * 
     * @param context Context onto which to create the builder
    */
    explicit SequenceBuilder(ContextHandle context);
    /**
     * @brief Creates a new SequenceBuilder
     * 
     * @param timeline Timeline used to synchronize steps
     * @param startValue Value the first step should wait for before starting
    */
    explicit SequenceBuilder(Timeline& timeline, uint64_t startValue = 0);
    ~SequenceBuilder();

private:
    struct pImp;
    std::unique_ptr<pImp> _pImp;
};

/**
 * @brief Creates a new SequenceBuilder
 * 
 * @param timeline Timeline used to synchronize steps
 * @param startValue Value the first step should wait for before starting
*/
[[nodiscard]]
inline SequenceBuilder beginSequence(Timeline& timeline, uint64_t startValue = 0) {
    return SequenceBuilder(timeline, startValue);
}
/**
 * @brief Creates a new SequenceBuilder
 * 
 * @note Creates an internal Timeline used for synchronizing steps
 * 
 * @param context Context onto which to create the builder
*/
[[nodiscard]]
inline SequenceBuilder beginSequence(const ContextHandle& context) {
    return SequenceBuilder(context);
}

/**
 * @brief Runs the given Command on the context and waits for it to finish
 * 
 * @param context Context to run the Command on
 * @param command Command to run
*/
HEPHAISTOS_API void execute(const ContextHandle& context, const Command& command);
/**
 * @brief Runs the given Subroutine on the context and wait for it to finish
 * 
 * @param context Context to run the Subroutine on
 * @param subroutine Subroutine to run
*/
HEPHAISTOS_API void execute(const ContextHandle& context, const Subroutine& subroutine);
/**
 * @brief Runs work on the given context and waits for it finish
 * 
 * @param context Context on which to run the work
 * @param emitter Function recording work
*/
HEPHAISTOS_API void execute(const ContextHandle& context,
    const std::function<void(vulkan::Command& cmd)>& emitter);
/**
 * @brief Runs the given sequence Command on the context
 * 
 * @param context Context on which to run the work
 * @param commands... Sequence of Commands to run
*/
template<std::derived_from<Command> ...T>
void executeList(const ContextHandle& context, const T& ...commands) {
    execute(context, [&commands...](vulkan::Command& cmd) {
        (commands.record(cmd), ...);
    });
}

}
