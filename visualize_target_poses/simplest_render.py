#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:25:59 2025

@author: ajay
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ✅ Define the MuJoCo model
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# ✅ Create the MuJoCo model and renderer
mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

# ✅ Initialize simulation (first frame)
mujoco.mj_forward(mj_model, mj_data)

# ✅ Render the first frame
renderer.update_scene(mj_data)  # Update scene before rendering
first_frame = renderer.render()  # Get pixel array

# ✅ Display using Matplotlib
plt.imshow(first_frame)
plt.axis("off")
plt.title("First Frame of MuJoCo Simulation")
plt.show()
