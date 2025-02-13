
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
import jax
import jax.numpy as jnp

def set_named_joints_qpos(model, qpos, joint_angles):
    """
    Updates qpos array using a dictionary of named joints and their target angles.

    Args:
        model: MjModel object (MuJoCo model).
        qpos: Initial qpos array.
        joint_angles: Dict of {joint_name: angle}

    Returns:
        Updated qpos array.
    """
    for joint_name, angle in joint_angles.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in the model!")

        qpos_index = model.jnt_qposadr[joint_id]
        if qpos_index >= len(qpos):
            raise ValueError(f"Invalid qpos index for joint '{joint_name}'")

        # ✅ Insert the angle into qpos array
        qpos = qpos.at[qpos_index].set(angle)

    return qpos
    


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
        """Resets the environment and sets a custom initial pose with named joints."""
        
        # Step 1️⃣: Initialize state from pipeline
        data = self.pipeline_init(self.sys.qpos0, jax.numpy.zeros(self.sys.nv))
        
        # Step 2️⃣: Set custom joint angles using names
        named_joint_angles = {
            "knee_left": 0.5,
            "knee_right": -0.5,
            #"hip_left": 0.1,
            #"hip_right": -0.1
        }
        
        # Step 3️⃣: Create updated qpos from named joints
        custom_qpos = set_named_joints_qpos(
            model=self.mj_model,
            qpos=self.sys.qpos0,
            joint_angles=named_joint_angles
        )
        
        # Step 4️⃣: Apply the pose using MJX-style immutable update
        data = data.replace(q=custom_qpos)
    
        # ✅ Debugging output (use jax.debug.callback for JIT compatibility)
        for joint_name in named_joint_angles.keys():
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_idx = self.mj_model.jnt_qposadr[joint_id]
        
            jax.debug.callback(
                lambda value, name=joint_name: print(f"{name} qpos: {value}"),
                custom_qpos[qpos_idx]
            )
    
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
