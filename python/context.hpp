#pragma once

#include <vector>

#include <hephaistos/handles.hpp>
#include <hephaistos/context.hpp>

const hephaistos::ContextHandle& getCurrentContext();
const std::vector<hephaistos::DeviceHandle>& getDevices();

void addExtension(hephaistos::ExtensionHandle extension, bool force);

void addResource(hephaistos::Resource& resource);
void removeResource(hephaistos::Resource& resource);
