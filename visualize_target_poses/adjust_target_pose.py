
"""
Created on Wed Feb 12 23:00:32 2025

@author: ajay
"""

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import matplotlib.pyplot as plt
from brax import envs
from brax.io import mjcf

# ✅ Define humanoid model path
HUMANOID_ROOT_PATH = "/home/ajay/Python_Projects/mujoco-mjx/mjx/mujoco/mjx/test_data/humanoid"

class Humanoid(envs.base.PipelineEnv):
    def __init__(self, exclude_current_positions_from_observation=True, **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(f"{HUMANOID_ROOT_PATH}/humanoid.xml")
        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)
        self.mj_model = mj_model
        self.renderer = mujoco.Renderer(mj_model)

        # ✅ Initialize missing attribute
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    def reset(self, rng: jax.numpy.ndarray):
        """Resets the environment to an initial state."""
        data = self.pipeline_init(self.sys.qpos0, jax.numpy.zeros(self.sys.nv))
        return envs.base.State(data, self._get_obs(data, jax.numpy.zeros(self.sys.nu)), 0.0, 0.0, {})

    def step(self, state, action):
        """Runs one timestep."""
        data = self.pipeline_step(state.pipeline_state, action)
        return state.replace(pipeline_state=data)

    def render_first_frame(self, rollout, camera_names):
        """Renders the first frame from each camera and saves as separate PNGs."""
        for cam_name in camera_names:
            frame = self.render([rollout[0]], camera=cam_name)  # Render first frame

            # ✅ Save each frame as a separate PNG
            plt.figure(figsize=(5, 5))
            plt.imshow(frame[0])
            plt.axis("off")
            plt.title(f"Camera: {cam_name}")
            plt.savefig(f"camera_{cam_name}.png", bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

    def _get_obs(
          self, data: mjx.Data, action: jp.ndarray
      ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
          position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])

# ✅ Register and create environment
envs.register_environment('humanoid', Humanoid)
env = envs.get_environment('humanoid')

# ✅ Define JIT-compiled reset and step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# ✅ Initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# ✅ Step the environment for 1 timestep
ctrl = -0.1 * jax.numpy.ones(env.sys.nu)
state = jit_step(state, ctrl)
rollout.append(state.pipeline_state)

# ✅ Render first frame and save each as a PNG
CAMERA_NAMES = ["back", "side", "front", "overhead"]
env.render_first_frame(rollout, CAMERA_NAMES)
