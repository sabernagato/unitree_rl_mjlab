#pragma once

// B2, B2-W, Go2 and Go2-W share the unitree_go low-level DDS messages.
#include "unitree/dds_wrapper/robots/go2/go2.h"

using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
