#pragma once

#include <vector>

#include "hephaistos/command.hpp"

namespace hephaistos {

class HEPHAISTOS_API StopWatch : public Resource {
public:
    [[nodiscard]] uint32_t getStopCount() const;

    [[nodiscard]] const Command& start();
    [[nodiscard]] const Command& stop();
    void reset();

    //stops not reached yet are identified via NaN
    //if wait = true, stops caller until all stops are available
    //time is in nanoseconds
    [[nodiscard]] std::vector<double> getTimeStamps(bool wait = false) const;

    StopWatch(const StopWatch&) = delete;
    StopWatch& operator=(const StopWatch&) = delete;
    
    StopWatch(StopWatch&& other) noexcept;
    StopWatch& operator=(StopWatch&& other) noexcept;

    StopWatch(ContextHandle context, uint32_t stops = 1);
    virtual ~StopWatch();

private:
    struct pImp;
    std::unique_ptr<pImp> _pImp;
};

}
