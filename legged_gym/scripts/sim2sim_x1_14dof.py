from legged_gym.envs import ZqSA01Cfg
from legged_gym.envs import x114DOF_Cfg
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs import *
from legged_gym.utils import Logger
import torch
import pygame
from threading import Thread

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.5, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = False
joystick_opened = False

if joystick_use:
    pygame.init()

    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False


    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd

        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = -joystick.get_axis(1) * x_vel_max
            y_vel_cmd = -joystick.get_axis(0) * y_vel_max
            yaw_vel_cmd = -joystick.get_axis(3) * yaw_vel_max

            pygame.time.delay(100)


    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


# This is the mapping from Isaac Gym joint order to MuJoCo joint order
# Format: isaac_gym_index: mujoco_index
ISAAC_TO_MUJOCO_MAP = {
    0: 3,  # left_hip_pitch_joint
    1: 4,  # left_hip_roll_joint
    2: 5,  # left_hip_yaw_joint
    3: 6,  # left_knee_pitch_joint
    4: 4,  # left_ankle_pitch_joint
    5: 8,  # left_ankle_roll_joint
    6: 1,  # left_shoulder_pitch_joint
    7: 9,  # right_hip_pitch_joint (was 8 in MuJoCo)
    8: 10,  # right_hip_roll_joint (was 9 in MuJoCo)
    9: 11,  # right_hip_yaw_joint (was 10 in MuJoCo)
    10: 12,  # right_knee_pitch_joint (was 11 in MuJoCo)
    11: 13,  # right_ankle_pitch_joint (was 12 in MuJoCo)
    12: 13,  # right_ankle_roll_joint (was 13 in MuJoCo)
    13: 7,  # right_shoulder_pitch_joint (was 7 in MuJoCo)
}

# The reverse mapping from MuJoCo to Isaac Gym
MUJOCO_TO_ISAAC_MAP = {v: k for k, v in ISAAC_TO_MUJOCO_MAP.items()}


def reorder_joints_mujoco_to_isaac(mujoco_array):
    """
    Reorder a MuJoCo joint array to match Isaac Gym joint order
    """
    isaac_array = np.zeros_like(mujoco_array)
    for mujoco_idx, isaac_idx in MUJOCO_TO_ISAAC_MAP.items():
        if mujoco_idx < len(mujoco_array) and isaac_idx < len(isaac_array):
            isaac_array[isaac_idx] = mujoco_array[mujoco_idx]
    return isaac_array


def reorder_joints_isaac_to_mujoco(isaac_array):
    """
    Reorder an Isaac Gym joint array to match MuJoCo joint order
    """
    mujoco_array = np.zeros_like(isaac_array)
    for isaac_idx, mujoco_idx in ISAAC_TO_MUJOCO_MAP.items():
        if isaac_idx < len(isaac_array) and mujoco_idx < len(mujoco_array):
            mujoco_array[mujoco_idx] = isaac_array[isaac_idx]
    return mujoco_array


def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('body-orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('body-angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = q[:3]

    # Extract joint positions and velocities then reorder to Isaac Gym convention
    joint_q = q[-model.nu:]
    joint_dq = dq[-model.nu:]

    # # Reorder from MuJoCo to Isaac Gym joint order
    # joint_q = reorder_joints_mujoco_to_isaac(joint_q)
    # joint_dq = reorder_joints_mujoco_to_isaac(joint_dq)

    # Extract foot positions and forces
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if 'ankle_roll' in body_name:
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double))

    # Debug foot positions
    # print(f"Foot positions: {foot_positions}")
    # print(f"Foot forces: {foot_forces}")

    return (joint_q, joint_dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd

    # Reorder torques from Isaac Gym to MuJoCo joint order for application
    torque_out_mujoco = reorder_joints_isaac_to_mujoco(torque_out)
    return torque_out_mujoco


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    model.opt.timestep = cfg.sim_config.dt

    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model

    # Use keyframe to set initial posture
    mujoco.mj_resetDataKeyframe(model, data, 0)  # Load first keyframe (default_pose)

    # Print keyframe information for debugging
    print(f"Loaded keyframe with qpos shape: {data.qpos.shape}")
    print(f"Initial joint positions: {data.qpos[-num_actuated_joints:]}")

    # Get default positions in MuJoCo order
    default_dof_pos_mujoco = reorder_joints_isaac_to_mujoco(cfg.robot_config.default_dof_pos)

    # Apply default positions in MuJoCo order
    data.qpos[-num_actuated_joints:] = default_dof_pos_mujoco

    mujoco.mj_step(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)

    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)

    stop_state_log = 4000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    # Print model actuation info for debugging
    print(f"Number of actuated joints: {num_actuated_joints}")
    print(f"Action space dimension: {cfg.env.num_actions}")
    print(f"Observation space dimension: {cfg.env.num_observations}")
    print(f"Single observation dimension: {cfg.env.num_single_obs}")

    # Print joint order from model
    print("Joint order in the model (MuJoCo):")
    for i in range(model.njnt):
        print(f"{i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")

    # Print actuator order from model
    print("Actuator order in the model (MuJoCo):")
    for i in range(model.nu):
        print(f"{i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

    # Print mapping between Isaac Gym and MuJoCo joints
    print("\nJoint mapping from Isaac Gym to MuJoCo:")
    for isaac_idx, mujoco_idx in ISAAC_TO_MUJOCO_MAP.items():
        isaac_name = [name for i, name in enumerate([
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "left_shoulder_pitch_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
            "right_hip_yaw_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint",
            "right_ankle_roll_joint", "right_shoulder_pitch_joint"
        ]) if i == isaac_idx][0]

        mujoco_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_idx)
        print(f"Isaac Gym {isaac_idx} ({isaac_name}) -> MuJoCo {mujoco_idx} ({mujoco_name})")

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):

        # Obtain an observation (already in Isaac Gym joint order)
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)

        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces

        # 1000hz -> 100hz (decimation)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # Create the observation vector
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # Fill in the observation vector
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
            obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel

            # Fill joint positions, velocities, and previous actions
            # q and dq are already in Isaac Gym order from get_obs
            obs[0, 5: 5 + cfg.env.num_actions] = (
                                                         q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 5 + cfg.env.num_actions: 5 + 2 * cfg.env.num_actions] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 5 + 2 * cfg.env.num_actions: 5 + 3 * cfg.env.num_actions] = action

            # Fill other state information
            obs[0, 5 + 3 * cfg.env.num_actions: 5 + 3 * cfg.env.num_actions + 3] = omega
            obs[0, 5 + 3 * cfg.env.num_actions + 3: 5 + 3 * cfg.env.num_actions + 6] = eu_ang

            # Clip observations
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            # Update the observation history
            hist_obs.append(obs)
            hist_obs.popleft()

            # Prepare policy input
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            # Get action from policy
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # Convert action to joint targets
            target_q = action * cfg.control.action_scale

        # Zero target velocities
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # Generate PD control - pd_control will handle the reordering to MuJoCo format
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques in MuJoCo order
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        # Apply control
        data.ctrl = tau

        # Get applied torques in MuJoCo order
        applied_tau = data.actuator_force

        # Convert applied torques back to Isaac Gym order for logging
        applied_tau_isaac = reorder_joints_mujoco_to_isaac(applied_tau)

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

        # Logging
        idx = 5  # Index for specific joint to log
        # Target positions are in Isaac Gym order
        dof_pos_target = target_q + cfg.robot_config.default_dof_pos

        if _ < stop_state_log:
            logger.log_states(
                {
                    'base_height': base_z,
                    'foot_z_l': foot_z[0] if len(foot_z) > 0 else 0.0,
                    'foot_z_r': foot_z[1] if len(foot_z) > 1 else 0.0,
                    'foot_forcez_l': foot_force_z[0] if len(foot_force_z) > 0 else 0.0,
                    'foot_forcez_r': foot_force_z[1] if len(foot_force_z) > 1 else 0.0,
                    'base_vel_x': v[0],
                    'command_x': x_vel_cmd,
                    'base_vel_y': v[1],
                    'command_y': y_vel_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': dof_pos_target[idx] if idx < len(dof_pos_target) else 0.0,
                    'dof_pos': q[idx] if idx < len(q) else 0.0,
                    'dof_vel': dq[idx] if idx < len(dq) else 0.0,
                    'dof_torque': applied_tau_isaac[idx] if idx < len(applied_tau_isaac) else 0.0,
                    'cmd_dof_torque': tau[ISAAC_TO_MUJOCO_MAP.get(idx, 0)] if idx < len(tau) else 0.0,
                    'dof_pos_target[0]': dof_pos_target[0].item() if 0 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[1]': dof_pos_target[1].item() if 1 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[2]': dof_pos_target[2].item() if 2 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[3]': dof_pos_target[3].item() if 3 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[4]': dof_pos_target[4].item() if 4 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[5]': dof_pos_target[5].item() if 5 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[6]': dof_pos_target[6].item() if 6 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[7]': dof_pos_target[7].item() if 7 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[8]': dof_pos_target[8].item() if 8 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[9]': dof_pos_target[9].item() if 9 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[10]': dof_pos_target[10].item() if 10 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[11]': dof_pos_target[11].item() if 11 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[12]': dof_pos_target[12].item() if 12 < len(dof_pos_target) else 0.0,
                    'dof_pos_target[13]': dof_pos_target[13].item() if 13 < len(dof_pos_target) else 0.0,
                    'dof_pos[0]': q[0].item() if 0 < len(q) else 0.0,
                    'dof_pos[1]': q[1].item() if 1 < len(q) else 0.0,
                    'dof_pos[2]': q[2].item() if 2 < len(q) else 0.0,
                    'dof_pos[3]': q[3].item() if 3 < len(q) else 0.0,
                    'dof_pos[4]': q[4].item() if 4 < len(q) else 0.0,
                    'dof_pos[5]': q[5].item() if 5 < len(q) else 0.0,
                    'dof_pos[6]': q[6].item() if 6 < len(q) else 0.0,
                    'dof_pos[7]': q[7].item() if 7 < len(q) else 0.0,
                    'dof_pos[8]': q[8].item() if 8 < len(q) else 0.0,
                    'dof_pos[9]': q[9].item() if 9 < len(q) else 0.0,
                    'dof_pos[10]': q[10].item() if 10 < len(q) else 0.0,
                    'dof_pos[11]': q[11].item() if 11 < len(q) else 0.0,
                    'dof_pos[12]': q[12].item() if 12 < len(q) else 0.0,
                    'dof_pos[13]': q[13].item() if 13 < len(q) else 0.0,
                    'dof_torque[0]': applied_tau_isaac[0].item() if 0 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[1]': applied_tau_isaac[1].item() if 1 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[2]': applied_tau_isaac[2].item() if 2 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[3]': applied_tau_isaac[3].item() if 3 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[4]': applied_tau_isaac[4].item() if 4 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[5]': applied_tau_isaac[5].item() if 5 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[6]': applied_tau_isaac[6].item() if 6 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[7]': applied_tau_isaac[7].item() if 7 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[8]': applied_tau_isaac[8].item() if 8 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[9]': applied_tau_isaac[9].item() if 9 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[10]': applied_tau_isaac[10].item() if 10 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[11]': applied_tau_isaac[11].item() if 11 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[12]': applied_tau_isaac[12].item() if 12 < len(applied_tau_isaac) else 0.0,
                    'dof_torque[13]': applied_tau_isaac[13].item() if 13 < len(applied_tau_isaac) else 0.0,
                    'dof_vel[0]': dq[0].item() if 0 < len(dq) else 0.0,
                    'dof_vel[1]': dq[1].item() if 1 < len(dq) else 0.0,
                    'dof_vel[2]': dq[2].item() if 2 < len(dq) else 0.0,
                    'dof_vel[3]': dq[3].item() if 3 < len(dq) else 0.0,
                    'dof_vel[4]': dq[4].item() if 4 < len(dq) else 0.0,
                    'dof_vel[5]': dq[5].item() if 5 < len(dq) else 0.0,
                    'dof_vel[6]': dq[6].item() if 6 < len(dq) else 0.0,
                    'dof_vel[7]': dq[7].item() if 7 < len(dq) else 0.0,
                    'dof_vel[8]': dq[8].item() if 8 < len(dq) else 0.0,
                    'dof_vel[9]': dq[9].item() if 9 < len(dq) else 0.0,
                    'dof_vel[10]': dq[10].item() if 10 < len(dq) else 0.0,
                    'dof_vel[11]': dq[11].item() if 11 < len(dq) else 0.0,
                    'dof_vel[12]': dq[12].item() if 12 < len(dq) else 0.0,
                    'dof_vel[13]': dq[13].item() if 13 < len(dq) else 0.0,
                }
            )

        elif _ == stop_state_log:
            logger.plot_states()

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg(x114DOF_Cfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x1_14DOF/mjcf/xyber_x1_flat.xml'
            sim_duration = 120.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # kps and kds in Isaac Gym order
            kps = np.array([60, 40, 35, 60, 35, 35, 20,
                            60, 40, 35, 60, 35, 35, 20], dtype=np.double)
            kds = np.array([3.0, 3.0, 4.0, 10.0, 3, 3, 3,
                            3.0, 3.0, 4.0, 10.0, 3, 3, 3], dtype=np.double)
            tau_limit = 500. * np.ones(14, dtype=np.double)
            # Default positions for all 14 joints in Isaac Gym order
            default_dof_pos = np.array([0.4, 0.05, -0.31, 0.48, -0.21, 0.0, 0.05,
                                        -0.4, -0.05, 0.31, 0.48, -0.21, 0.0, -0.05])


    # Verify the model can be loaded
    try:
        model_path = Sim2simCfg.sim_config.mujoco_model_path
        print(f"Loading model from: {model_path}")
        model = mujoco.MjModel.from_xml_path(model_path)
        print(f"Model loaded successfully with {model.nq} generalized coordinates (qpos) and {model.nu} actuators")
    except Exception as e:
        print(f"Error loading model: {e}")
        import sys

        sys.exit(1)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())