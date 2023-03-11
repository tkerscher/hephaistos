#pragma once

#include <hephaistos/handles.hpp>

const hephaistos::ContextHandle& getCurrentContext();

struct CommandHandle {
    hephaistos::CommandHandle command;
};

