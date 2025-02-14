"""
Custom environment

Humanoid one leg balance task

"""


def compute_com(model, data):
    """
    Computes the mass-weighted Center of Mass (CoM).
    """    
    mujoco.mj_forward(model, data)  # Ensure xpos is updated
    
    masses = model.body_mass  # Masses of all bodies
    body_positions = data.xpos  # Global positions of all bodies

    # Compute mass-weighted CoM
    com = jnp.sum(masses[:, None] * body_positions, axis=0) / jnp.sum(masses)
    return com


# example useage
#com = compute_com(model, data)



def get_foot_position(model, data, foot_name="foot_left"):
    """
    Extracts the position of the chosen support foot.
    """
    foot_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
    foot_position = data.xpos[foot_idx]  # Extract (x, y, z) position
    return foot_position

# example useage
#foot_pos = get_foot_position(model, data)


# %% Balance Cost: Keep CoM Above the Foot

"""
We penalize the horizontal displacement of the CoM from the support foot.

J_balance = max(0,d_com - d_max)^2

where 

d_com is the horizontal distance between CoM and foot

d_max is the allowed threshold (foot size)

"""

def balance_cost(model, data, foot_position, max_distance=0.1):
    """
    Penalizes CoM if it moves too far from the foot.
    """
    mujoco.mj_forward(model, data)  # Ensure xpos is updated

    # Compute CoM
    com_position = compute_com(model, data)

    # Compute horizontal CoM deviation (x, y only)
    com_dist = jnp.linalg.norm(com_position[:2] - foot_position[:2])

    # Quadratic penalty if out of bounds
    return jnp.maximum(0, com_dist - max_distance) ** 2

# %% Posture Regularization: Keep Joint Angles in a Natural Range

# We define a neutral pose (e.g., standing upright) and penalize deviation.

def posture_cost(data, joint_weights=None):
    """
    Penalizes unnatural joint angles.
    """
    # Neutral standing pose (adjust values as needed)
    neutral_posture = jnp.zeros_like(data.qpos[7:])  # Exclude root position

    # Joint weight vector (if None, use equal weights)
    if joint_weights is None:
        joint_weights = jnp.ones_like(neutral_posture)

    # Compute cost
    return jnp.sum(joint_weights * (data.qpos[7:] - neutral_posture) ** 2)

# %% Effort Cost: Minimize Control Torques

# We minimize the sum of squared joint torques to avoid excessive energy use

def effort_cost(data, torque_weights=None):
    """
    Penalizes high joint torques.
    """
    # Torque weight vector (if None, use equal weights)
    if torque_weights is None:
        torque_weights = jnp.ones_like(data.ctrl)

    # Compute cost
    return jnp.sum(torque_weights * data.ctrl ** 2)

# %% Smoothenss cost

def smoothness_cost(data, prev_action, smoothness_weights=None):
    """
    Penalizes large changes in control actions.
    """
    if smoothness_weights is None:
        smoothness_weights = jnp.ones_like(data.ctrl)

    return jnp.sum(smoothness_weights * (data.ctrl - prev_action) ** 2)


# %% Full MPC Cost Function

def mpc_cost(model, data, prev_action, foot_position):
    """
    Computes the full MPC cost function.
    """
    # Define weights
    w_balance = 1.0
    w_posture = 0.1
    w_effort = 0.01
    w_smooth = 0.05

    # Compute individual cost terms
    balance_term = balance_cost(model, data, foot_position)
    posture_term = posture_cost(data)
    effort_term = effort_cost(data)
    smooth_term = smoothness_cost(data, prev_action)

    # Final cost function
    total_cost = (w_balance * balance_term +
                  w_posture * posture_term +
                  w_effort * effort_term +
                  w_smooth * smooth_term)
    
    return total_cost

# %%

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import State
from brax import envs as brax_envs
from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
import dial_mpc.envs as dial_envs
from brax.io import mjcf
from brax.physics.base import System

@dataclass
class MyEnvConfig(BaseEnvConfig):
    arg1: float = 1.0
    arg2: str = "test"
    timestep: float = 0.005  # Default MuJoCo timestep

class MyEnv(BaseEnv):
    def __init__(self, config: MyEnvConfig):
        super().__init__(config)
        self.model_path = "my_model/my_model.xml"
        self.sys = self.make_system(config)
        self.data = mujoco.MjData(self.sys)

    def make_system(self, config: MyEnvConfig) -> System:
        """
        Loads the MuJoCo system from MJCF and applies the timestep.
        """
        sys = mujoco.MjModel.from_xml_path(self.model_path)
        sys.opt.timestep = config.timestep  # Set timestep from config
        return sys

    def reset(self, rng: jax.Array) -> State:
        """
        Resets the MuJoCo simulation and returns the initial state.
        """
        mujoco.mj_resetData(self.sys, self.data)  # Reset simulation state
        mujoco.mj_forward(self.sys, self.data)  # Compute initial state
        
        # Extract state information
        qpos = jnp.array(self.data.qpos)  # Joint positions
        qvel = jnp.array(self.data.qvel)  # Joint velocities

        # Return Brax-compatible state
        return State(qp=qpos, obs=qpos, reward=0.0, done=False, info={})

    def step(self, state: State, action: jax.Array) -> State:
        """
        Applies action, steps the simulation, and returns the next state.
        """
        # Apply control action (torques)
        self.data.ctrl[:] = jnp.array(action)

        # Step the simulation
        mujoco.mj_step(self.sys, self.data)

        # Extract next state information
        qpos = jnp.array(self.data.qpos)
        qvel = jnp.array(self.data.qvel)

        # Compute the cost (negative reward)
        foot_position = self.data.xpos[self.sys.body("foot_left").id]  # Example: left foot
        cost = balance_cost(self.sys, self.data, foot_position)

        # Compute termination condition (e.g., if humanoid falls)
        done = qpos[2] < 0.4  # If torso height drops below 0.4m, terminate

        # Return next state
        return State(qp=qpos, obs=qpos, reward=-cost, done=done, info={})

# Register the environment
brax_envs.register_environment("my_env_name", MyEnv)
dial_envs.register_config("my_env_name", MyEnvConfig)


