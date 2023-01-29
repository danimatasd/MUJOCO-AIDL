#Enviroment Settings:
#sudo apt install ffmpeg
#git clone https://github.com/deepmind/mujoco
#env MUJOCO_GL=egl

#@title Other imports and helper functions
import mujoco
import numpy as np
import os

from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
 
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

print('Get current working directory : ', os.getcwd())

model = mujoco.MjModel.from_xml_path("./anybotics_anymal_c/scene.xml")
data = mujoco.MjData(model)
renderer=mujoco.Renderer(model)

mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)

print(f"Timestep: {model.opt.timestep}")

DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 3

timevals = []
q_velocity = []
x_position = []
frames = []

# Simulate and save data
mujoco.mj_resetDataKeyframe(model, data, 0)

while data.time < DURATION:
  # Set control vector.
  movmtrx = np.identity(12)
  rbsop = movmtrx [0,:]
  rbsrot = movmtrx [1,:]
  rbkrot = movmtrx [2,:]
  lbsop = movmtrx [3,:]
  lbsrot = movmtrx [4,:]
  lbkrot = movmtrx [5,:]
  rfsop = movmtrx [6,:]
  rfsrot = movmtrx [7,:]
  rfkrot = movmtrx [8,:]
  lfsop = movmtrx [9,:]
  lfsrot = movmtrx [10,:]
  lfkrot = movmtrx [11,:]
  data.ctrl = (lfkrot+rfkrot+lfsrot+rfsrot)*np.sin(data.time)*(-0.2)+(lbkrot+rbkrot+lbsrot+rbsrot)*np.cos(data.time*2)*(-0.2)
  
  # Step the simulation.
  mujoco.mj_step(model, data)
  
  timevals.append(data.time)
  q_velocity.append(data.qvel.copy())
  x_position.append(data.geom_xpos[2,2])

  # Render and save frames.
  if len(frames) < data.time * FRAMERATE:
    # Set the lookat point to the humanoid's center of mass.
    camera.lookat = data.body('LH_SHANK').subtree_com

    renderer.update_scene(data, camera)
    pixels = renderer.render()
    frames.append(pixels.copy())

media.show_video(frames, fps=FRAMERATE)

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, q_velocity)
ax[0].set_title('q velocity')
ax[0].set_ylabel('velocity')

ax[1].plot(timevals, x_position)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('position')
_ = ax[1].set_title('x position')