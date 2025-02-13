"""

See - https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

Use conda env mujoco-mjx

"""

import os

# Ensure GLFW is used instead of OSMesa if possible - must be set before importing dm_control
os.environ["MUJOCO_GL"] = "glfw"

import jax
print(jax.devices())

# if the print out is not [CudaDevice(id=0)] need to restart machine !!!
# %%

#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
#print('Installing mediapy:')
#command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
#pip install mediapy
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
import os

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

# %% Defnine Humanoid Env

import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jp
from etils import epath
from brax.envs.base import PipelineEnv, State
from brax import envs

# Define environment path
HUMANOID_ROOT_PATH = epath.Path('/home/ajay/Python_Projects/mujoco-mjx/mjx/mujoco/mjx/test_data/humanoid')

class HumanoidBalance(PipelineEnv):
    """Humanoid learns to stand on one leg while maintaining the existing health check logic."""

    def __init__(
        self,
        pose_reward_weight=0.0, # this is too artificial and restrictive for a heel-to-toe balance
        stability_reward_weight=3.0,
        ctrl_cost_weight=0.1,
        non_standing_foot_penalty_weight=1.0,  # NEW penalty weight
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        # Load Mujoco Model
        mj_model = mujoco.MjModel.from_xml_path(
            (HUMANOID_ROOT_PATH / 'humanoid.xml').as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        # Store environment parameters
        self._pose_reward_weight = pose_reward_weight
        self._stability_reward_weight = stability_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._non_standing_foot_penalty_weight = non_standing_foot_penalty_weight  # NEW
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # # Define target pose for standing on one leg - this needs to be modified for the heel_to_toe task
        # self._target_qpos = jp.array([
        #     0, 0, 1.21948, 0.971588, -0.179973, 0.135318, -0.0729076,
        #     -0.0516, -0.202, 0.23, -0.24, -0.007, -0.34, -1.76, -0.466, -0.0415,
        #     -0.08, -0.01, -0.37, -0.685, -0.35, -0.09, 0.109, -0.067, -0.7, -0.05, 0.12, 0.16
        # ])

        # ✅ Correctly Retrieve Torso Index
        self._torso_index = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        assert self._torso_index != -1, "Error: 'torso' body not found in model!"

        # ✅ Correctly Get Foot Indices Using `mj_name2id`
        self._left_foot_index = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "foot_left")
        self._right_foot_index = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")

        # Ensure indices are valid
        assert self._left_foot_index != -1, "Error: 'foot_left' not found in model!"
        assert self._right_foot_index != -1, "Error: 'foot_right' not found in model!"


    #===================================================================================================    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
    
        # ✅ New Stability Penalty (CoM should be near either foot)
        com_xy = data.subtree_com[1][:2]  # CoM position in x-y plane
        foot_left_xy = data.xpos[self._left_foot_index][:2]  # Left foot position in x-y
        foot_right_xy = data.xpos[self._right_foot_index][:2]  # Right foot position in x-y
        
        # Compute distance from CoM to the closest foot
        dist_left = jp.linalg.norm(com_xy - foot_left_xy)
        dist_right = jp.linalg.norm(com_xy - foot_right_xy)
        min_dist = jp.minimum(dist_left, dist_right)  # Distance to the nearest foot
        
        # Define a "safe" distance where no penalty applies
        safe_radius = 0.15  # The CoM can move within 15 cm freely
        
        # Apply quadratic penalty if CoM is outside the safe zone
        stability_penalty = jp.where(
            min_dist > safe_radius,
            (min_dist - safe_radius) ** 2,  # Quadratic penalty
            0.0  # No penalty if CoM is within the safe region
        )
        
        # ✅ Forward Motion Reward (Encourage Walking)
        forward_velocity = data.qvel[0]  # Extract x-axis velocity
        forward_motion_reward = jp.maximum(0, forward_velocity) # Reward forward movement
    
        # ✅ New: **Foot Orientation Penalty** (Encourages feet to point forward)
        torso_quat = data.xquat[self._torso_index]  # Torso quaternion
        foot_left_quat = data.xquat[self._left_foot_index]  # Left foot quaternion
        foot_right_quat = data.xquat[self._right_foot_index]  # Right foot quaternion
        
        def forward_vector_from_quaternion(q):
            """Extracts the forward direction from a quaternion (q)."""
            w, x, y, z = q
            return jp.array([
                1 - 2 * (y ** 2 + z ** 2),  # x-direction (forward)
                2 * (x * y + w * z),        # y-direction (side)
                2 * (x * z - w * y)         # z-direction (up/down)
            ])
    
        # Compute forward vectors for torso and feet
        torso_forward = forward_vector_from_quaternion(torso_quat)
        foot_left_forward = forward_vector_from_quaternion(foot_left_quat)
        foot_right_forward = forward_vector_from_quaternion(foot_right_quat)
        
        # Compute alignment error (dot product measures similarity)
        left_alignment = jp.dot(torso_forward, foot_left_forward)
        right_alignment = jp.dot(torso_forward, foot_right_forward)
    
        # Define penalty for feet not pointing forward
        foot_orientation_penalty = (
            (1.0 - left_alignment) ** 2
            + (1.0 - right_alignment) ** 2
        )
    
        # ✅ New: Foot Alignment Penalty (Ensures both feet stay parallel)
        foot_alignment_penalty = (1.0 - jp.dot(foot_left_forward, foot_right_forward)) ** 2
    
        # ✅ New: Straight Line Walking Penalty (Feet should remain on the same y-axis)
        foot_left_pos = data.xpos[self._left_foot_index]  # Left foot position
        foot_right_pos = data.xpos[self._right_foot_index]  # Right foot position
    
        foot_y_deviation = jp.abs(foot_left_pos[1] - foot_right_pos[1])  # Difference in lateral (y) position
        foot_straightness_penalty = foot_y_deviation ** 2  # Quadratic penalty for deviation
    
        # ✅ Healthy Reward
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = self._healthy_reward * is_healthy if not self._terminate_when_unhealthy else self._healthy_reward
    
        # ✅ Control Cost Penalty
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    
        # ✅ Final Reward
        reward = (
            #self._stability_reward_weight * stability_reward
            forward_motion_reward
            - ctrl_cost
            - foot_orientation_penalty  # ✅ Penalizes feet not pointing forward
            - foot_alignment_penalty  # ✅ Penalizes feet not being parallel
            - foot_straightness_penalty  # ✅ Penalizes feet not staying in a straight line
            - stability_penalty  # ✅ Added here
            + healthy_reward
        )
    
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
        state.metrics.update(
            forward_motion_reward=forward_motion_reward,
            ctrl_cost=ctrl_cost,
            foot_orientation_penalty=-foot_orientation_penalty,
            foot_alignment_penalty=-foot_alignment_penalty,  # NEW
            foot_straightness_penalty=-foot_straightness_penalty,  # NEW
            stability_penalty=-stability_penalty,  # ✅ Added metric
            reward_alive=healthy_reward,
            )
    
        return state.replace(pipeline_state=data, obs=self._get_obs(data, action), reward=reward, done=done)
    
    #===================================================================================================    

    # def reset(self, rng: jp.ndarray) -> State:
    #     """Resets the environment to an initial state."""
    #     rng, rng1, rng2 = jax.random.split(rng, 3)
    
    #     low, hi = -self._reset_noise_scale, self._reset_noise_scale
    #     qpos = self.sys.qpos0 + jax.random.uniform(
    #         rng1, (self.sys.nq,), minval=low, maxval=hi
    #     )
    #     qvel = jax.random.uniform(
    #         rng2, (self.sys.nv,), minval=low, maxval=hi
    #     )
    
    #     data = self.pipeline_init(qpos, qvel)
    
    #     # ✅ Correctly Retrieve Torso Index
    #     self._torso_index = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    #     assert self._torso_index != -1, "Error: 'torso' body not found in model!"
    
    #     # ✅ Healthy Reward Calculation
    #     min_z, max_z = self._healthy_z_range
    #     is_healthy = jp.where(qpos[2] < min_z, 0.0, 1.0)
    #     is_healthy = jp.where(qpos[2] > max_z, 0.0, is_healthy)
    #     healthy_reward = self._healthy_reward * is_healthy if not self._terminate_when_unhealthy else self._healthy_reward
    
    #     done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    #     obs = self._get_obs(data, jp.zeros(self.sys.nu))
    
    #     # ✅ Ensure proper JAX structure consistency
    #     batch_size = qpos.shape[0]  # Ensure we match the batch size of `step()`
    #     zero = jp.zeros((batch_size,))  # ✅ Ensures all initialized values have batch dimensions

    
    #     # ✅ **Initialize All Metrics to Zero (Simplified Code)**
    #     metrics = {
    #         'forward_motion_reward': zero,
    #         'ctrl_cost': zero,
    #         'stability_penalty': zero,
    #         'foot_orientation_penalty': zero,
    #         'foot_alignment_penalty': zero,
    #         'foot_straightness_penalty': zero,
    #         'reward_alive': jp.full((batch_size,), healthy_reward),  # ✅ Matches batch dimension
    #     }
    
    #     return State(data, obs, zero, done, metrics)

    #===================================================================================================    

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
    
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )
    
        data = self.pipeline_init(qpos, qvel)
    
        # ✅ Correctly Retrieve Torso Index
        self._torso_index = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        assert self._torso_index != -1, "Error: 'torso' body not found in model!"
    
        # ✅ New Stability Penalty (CoM should be near either foot)
        com_xy = data.subtree_com[1][:2]  # CoM position in x-y plane
        foot_left_xy = data.xpos[self._left_foot_index][:2]  # Left foot position in x-y
        foot_right_xy = data.xpos[self._right_foot_index][:2]  # Right foot position in x-y
    
        # Compute distance from CoM to the closest foot
        dist_left = jp.linalg.norm(com_xy - foot_left_xy)
        dist_right = jp.linalg.norm(com_xy - foot_right_xy)
        min_dist = jp.minimum(dist_left, dist_right)  # Distance to the nearest foot
    
        # Define a "safe" distance where no penalty applies
        safe_radius = 0.15  # The CoM can move within 15 cm freely
    
        # Apply quadratic penalty if CoM is outside the safe zone
        stability_penalty = jp.where(
            min_dist > safe_radius,
            (min_dist - safe_radius) ** 2,  # Quadratic penalty
            0.0  # No penalty if CoM is within the safe region
        )
    
        # ✅ New: **Foot Orientation Penalty** at Reset (Encourages feet to point forward)
        torso_quat = data.xquat[self._torso_index]  # Torso quaternion
        foot_left_quat = data.xquat[self._left_foot_index]  # Left foot quaternion
        foot_right_quat = data.xquat[self._right_foot_index]  # Right foot quaternion
    
        def forward_vector_from_quaternion(q):
            """Extracts the forward direction from a quaternion (q)."""
            w, x, y, z = q
            return jp.array([
                1 - 2 * (y ** 2 + z ** 2),  # x-direction (forward)
                2 * (x * y + w * z),        # y-direction (side)
                2 * (x * z - w * y)         # z-direction (up/down)
            ])
    
        # Compute forward vectors for torso and feet
        torso_forward = forward_vector_from_quaternion(torso_quat)
        foot_left_forward = forward_vector_from_quaternion(foot_left_quat)
        foot_right_forward = forward_vector_from_quaternion(foot_right_quat)
    
        # Compute alignment (dot product measures similarity)
        left_alignment = jp.dot(torso_forward, foot_left_forward)
        right_alignment = jp.dot(torso_forward, foot_right_forward)
    
        # Define penalty for feet not pointing forward
        foot_orientation_penalty = (
            (1.0 - left_alignment) ** 2
            + (1.0 - right_alignment) ** 2
        )
    
        # ✅ New: Foot Alignment Penalty (Ensures both feet stay parallel)
        foot_alignment_penalty = (1.0 - jp.dot(foot_left_forward, foot_right_forward)) ** 2
    
        # ✅ New: Straight Line Walking Penalty (Feet should remain on the same y-axis)
        foot_y_deviation = jp.abs(foot_left_xy[1] - foot_right_xy[1])  # Difference in lateral (y) position
        foot_straightness_penalty = foot_y_deviation ** 2  # Quadratic penalty for deviation
    
        # ✅ **Forward Motion Reward** (Encourages Walking)
        forward_velocity = data.qvel[0]  # Extract x-axis velocity
        forward_motion_reward = jp.maximum(0, forward_velocity) * 2.0  # Reward forward movement
    
        # ✅ Healthy Reward (same as step function)
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qpos[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(qpos[2] > max_z, 0.0, is_healthy)
        healthy_reward = self._healthy_reward * is_healthy if not self._terminate_when_unhealthy else self._healthy_reward
    
        # ✅ Control cost is 0 at reset since no action taken
        ctrl_cost = 0.0
    
        # ✅ Compute total initial reward
        reward = (
            forward_motion_reward  # ✅ Encourages forward movement
            - stability_penalty  # ✅ Penalizes CoM being too far from both feet
            - 2.0 * foot_orientation_penalty  # ✅ Penalizes feet not pointing forward
            - foot_alignment_penalty  # ✅ Penalizes feet not being parallel
            - foot_straightness_penalty  # ✅ Penalizes feet not staying in a straight line
            + healthy_reward
        )
    
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
        obs = self._get_obs(data, jp.zeros(self.sys.nu))
    
        # ✅ Initialize ALL metrics to zero to ensure JAX structure consistency
        metrics = {
            'forward_motion_reward': forward_motion_reward,  # ✅ Encourages forward movement,
            'ctrl_cost': -ctrl_cost,
            'stability_penalty': -stability_penalty,
            'foot_orientation_penalty': -foot_orientation_penalty,
            'foot_alignment_penalty': -foot_alignment_penalty,
            'foot_straightness_penalty': -foot_straightness_penalty,
            'reward_alive': healthy_reward,
        }
    
        return State(data, obs, reward, done, metrics)

  #===================================================================================================       

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and actuator forces."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]  # Remove global x, y coordinates
    
        # ✅ Concatenate all observation components
        return jp.concatenate([
            position,                # Joint positions
            data.qvel,               # Joint velocities
            data.cinert[1:].ravel(), # Center of mass inertia
            data.cvel[1:].ravel(),   # Center of mass velocity
            data.qfrc_actuator,      # Actuator forces
            action                   # Last action taken
        ])

envs.register_environment('humanoid_balance', HumanoidBalance)

# %%  instantiate the environment

env_name = 'humanoid_balance'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# %%  initialize the state

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
for i in range(10):
  ctrl = -0.1 * jp.ones(env.sys.nu)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

#media.show_video(env.render(rollout, camera='side'), fps=1.0 / env.dt)

# Render frames from the environment
frames = env.render(rollout, camera='side')

# Save as an MP4 file
video_path = "/home/ajay/Python_Projects/mujoco-mjx/videos/balance_task_rollout.mp4"
media.write_video(video_path, frames, fps=1.0 / env.dt)

# %% Define training function

from brax.training.agents.ppo import train as ppo
#from brax.training.agents.ppo import networks as ppo_networks

train_fn = functools.partial(
    ppo.train, 
    num_timesteps=200_000_000, 
    num_evals=5, 
    reward_scaling=1.0, # changed from 0.1 to 1.0
    episode_length=1000, 
    normalize_observations=True, 
    action_repeat=1,
    unroll_length=10, 
    num_minibatches=24, 
    num_updates_per_batch=8,
    discounting=0.97, 
    learning_rate=3e-4, 
    entropy_cost=1e-3, 
    num_envs=3072,
    batch_size=512, 
    seed=0)

# %% Start training

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0
def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])

  plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
  plt.ylim([min_y, max_y])

  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')

  plt.errorbar(
      x_data, y_data, yerr=ydataerr)
  plt.show()

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# %% Save Model

model_path = '/home/ajay/Python_Projects/mujoco-mjx/saved_models/mjx_brax_policy'
model.save_params(model_path, params)

# %% Load Model and Define Inference Function

params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# %% Visualize Policy

eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(2)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

  if state.done:
    break

#media.show_video(env.render(rollout[::render_every], camera='side'), fps=1.0 / env.dt / render_every)

# %% Visualize

import mediapy as media

# Generate frames - side view
frames = env.render(rollout[::render_every], camera='side')
# Define video file path
video_path = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_side_view.mp4"
# Save frames as an MP4 video
media.write_video(video_path, frames, fps=1.0 / env.dt / render_every)
print(f"Video saved as: {video_path}")

# Generate frames - back view
frames_back_view = env.render(rollout[::render_every], camera='back')
# Define video file path
video_path_back_view = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_back_view.mp4"
# Save frames as an MP4 video
media.write_video(video_path_back_view, frames_back_view, fps=1.0 / env.dt / render_every)
print(f"Video saved as: {video_path_back_view}")

# Generate frames - overhead view
frames_overhead_view = env.render(rollout[::render_every], camera='overhead')
# Define video file path
video_path_overhead_view = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_overhead_view.mp4"
# Save frames as an MP4 video
media.write_video(video_path_overhead_view, frames_overhead_view, fps=1.0 / env.dt / render_every)
print(f"Video saved as: {video_path_overhead_view}")

# Generate frames - front view
frames_front_view = env.render(rollout[::render_every], camera='front')
# Define video file path
video_path_front_view = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_front_view.mp4"
# Save frames as an MP4 video
media.write_video(video_path_front_view, frames_front_view, fps=1.0 / env.dt / render_every)
print(f"Video saved as: {video_path_front_view}")

# %% Convert MP4 to GIF

import os
import imageio
import numpy as np
from PIL import Image

def mp4_to_gif(mp4_path, gif_path, fps=20, scale=480):
    """Converts an MP4 video to a GIF."""
    reader = imageio.get_reader(mp4_path, 'ffmpeg')
    writer = imageio.get_writer(gif_path, fps=fps)

    for i, frame in enumerate(reader):
        img = Image.fromarray(frame)
        img = img.resize((scale, int(img.height * scale / img.width)), Image.LANCZOS)
        writer.append_data(np.array(img))
    
    writer.close()
    print(f"✅ GIF saved at: {gif_path}")

# Example usage:
mp4_file = video_path #"input.mp4"  # Change this to your MP4 file
gif_file = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_side.gif"
mp4_to_gif(mp4_file, gif_file, fps=20, scale=480)

mp4_file = video_path_back_view #"input.mp4"  # Change this to your MP4 file
gif_file = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_back.gif"
mp4_to_gif(mp4_file, gif_file, fps=20, scale=480)

mp4_file = video_path_overhead_view #"input.mp4"  # Change this to your MP4 file
gif_file = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_overhead.gif"
mp4_to_gif(mp4_file, gif_file, fps=20, scale=480)

mp4_file = video_path_front_view #"input.mp4"  # Change this to your MP4 file
gif_file = "/home/ajay/Python_Projects/mujoco-mjx/videos/trainned_PPO_policy_front.gif"
mp4_to_gif(mp4_file, gif_file, fps=20, scale=480)


