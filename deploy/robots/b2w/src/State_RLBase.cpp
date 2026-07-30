#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <array>
#include <stdexcept>

namespace
{
// Policy leg order is FL, FR, RL, RR. Unitree LowCmd order is
// FR, FL, RR, RL, followed by FR, FL, RR, RL wheel motors.
constexpr std::array<int, 12> kLegActionToSdk = {
    3, 4, 5,
    0, 1, 2,
    9, 10, 11,
    6, 7, 8,
};
constexpr std::array<int, 4> kWheelActionToSdk = {13, 12, 15, 14};
constexpr float kPosStopF = 2.146e9F;
constexpr std::size_t kActionDim = 16;
}  // namespace

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    if (env->robot->data.joint_ids_map.size() != kActionDim ||
        env->action_manager->total_action_dim() != static_cast<int>(kActionDim))
    {
        throw std::runtime_error("B2-W deploy configuration must contain 16 state joints and 16 actions.");
    }

    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    const auto action = env->action_manager->processed_actions();
    if (action.size() != kActionDim)
    {
        return;
    }

    for (std::size_t i = 0; i < kLegActionToSdk.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[kLegActionToSdk[i]];
        motor.q() = action[i];
        motor.dq() = 0.0F;
        motor.tau() = 0.0F;
    }

    for (std::size_t i = 0; i < kWheelActionToSdk.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[kWheelActionToSdk[i]];
        motor.q() = kPosStopF;
        motor.kp() = 0.0F;
        motor.dq() = action[kLegActionToSdk.size() + i];
        motor.tau() = 0.0F;
    }
}
