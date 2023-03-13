#pragma once

#include <vector>

#include <hephaistos/handles.hpp>

const hephaistos::ContextHandle& getCurrentContext();
const std::vector<hephaistos::DeviceHandle>& getDevices();
