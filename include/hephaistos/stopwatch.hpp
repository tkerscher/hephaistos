#pragma once

#include <vector>

#include "hephaistos/command.hpp"

namespace hephaistos {

/**
 * @brief Allows measuring of elapsed time between commands execution
 * 
 * Stopwatch allows to record timestamps on the device's pipeline, thus
 * circumventing the synchronization overhead needed if one takes them on the
 * host instead. The first timestamp is recorded once previous commands are
 * issued onto the pipeline, the second once they leave the pipeline, i.e. are
 * finished, allowing to approximately calculate the elapsed time during the
 * execution of the commands between. Due to the nature of device's pipeline
 * and its internal precision this is not an exact method, but should produce
 * more accurate measurements than methods on the host.
*/
class HEPHAISTOS_API StopWatch : public Resource {
public:
    /**
     * @brief Returns the command for starting the stop watch.
     * 
     * Issues the device to save a timestamp once previous commands have been
     * issued onto the device's pipeline and ensures later commands are only
     * processed once the timestamp has been saved.
     * 
     * @return Command for starting the stop watch.
    */
    [[nodiscard]] const Command& start();
    /**
     * @brief Returns the command for recording a timestamp.
     * 
     * Issues the device to save a timestamp once previous commands have left
     * the pipeline, i.e. are finished.
     * 
     * @return Command for recording a timestamp.
    */
    [[nodiscard]] const Command& stop();
    /**
     * @brief Resets the Stopwatch
     * 
     * 
    */
    void reset();

    /**
     * @brief Returns the elapsed time between start() and stop() in nanoseconds
     * 
     * Calculates the elapsed time between the timestamps the device recorded
     * during its execution of the start() end stop() command in nanoseconds.
     * If wait is true, blocks the call until both timestamps are recorded,
     * otherwise returns NaN if they are not yet available.
     * 
     * @param wait If true, blocks the calling code until all timestamps are
     *             recorded.
     * 
     * @return Elapsed time between timestamps in nanoseconds
    */
    [[nodiscard]] double getElapsedTime(bool wait = false) const;

    StopWatch(const StopWatch&) = delete;
    StopWatch& operator=(const StopWatch&) = delete;
    
    StopWatch(StopWatch&& other) noexcept;
    StopWatch& operator=(StopWatch&& other) noexcept;

    /**
     * @brief Creates a new StopWatch on the given context.
     * 
     * @param context Context used to create the StopWatch
    */
    explicit StopWatch(ContextHandle context);
    ~StopWatch() override;

private:
    struct pImp;
    std::unique_ptr<pImp> _pImp;
};

}
