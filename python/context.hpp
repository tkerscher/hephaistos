#pragma once

#include <vector>

#include <hephaistos/handles.hpp>
#include <hephaistos/context.hpp>

const hephaistos::ContextHandle& getCurrentContext();
const std::vector<hephaistos::DeviceHandle>& getDevices();
const hephaistos::DeviceHandle& getDevice(uint32_t id);

void addExtension(hephaistos::ExtensionHandle extension, bool force);
