#Enviroment Settings:
#sudo apt install ffmpeg
#git clone https://github.com/deepmind/mujoco
#env MUJOCO_GL=egl
#./../../Mujoco/bin/simulate ./anybotics_anymal_c/scene.xml

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

model = mujoco.MjModel.from_xml_path("./anybotics_anymal_c/scene2.xml")
data = mujoco.MjData(model)
renderer=mujoco.Renderer(model)

mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)

print(f"Timestep: {model.opt.timestep}")

DURATION  = 4   # seconds
FRAMERATE = 60  # Hz

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 6

timevals = []
q_position = []
q_velocity = []
sensordata = []
frames = []

'''mujoco.mj_step(model, data)
q_position.append(data.qpos.copy())
print(q_position)'''

# Simulate and save data
mujoco.mj_resetDataKeyframe(model, data, 0)

while data.time < DURATION:
  # Set control vector joint aproach
  '''movmtrx = np.identity(12)
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
  if data.time < 1: data.ctrl = (lfkrot+rfkrot+lfsrot+rfsrot)*np.sin(data.time)*(-0.2)+(lbkrot+rbkrot+lbsrot+rbsrot)*np.cos(data.time*2)*(-0.2)
  else: data.ctrl = 0'''
  
  # Set control vector array aproach
  ctrl=np.zeros(12)
  for i in range(4):
    #ctrl[(i*3)]=1
    ctrl[(i*3)+1]=1
  if data.time < 5: data.ctrl = ctrl*(-1)
  else: data.ctrl = 0
  
  # Step the simulation.
  mujoco.mj_step(model, data)
  
  timevals.append(data.time)
  q_position.append(data.qpos[2].copy())
  #print(data.qpos[2])
  q_velocity.append(data.qvel[8:].copy())
  #sensordata.append(data.sensor('velocimeter').data.copy())

  # Render and save frames.
  if len(frames) < data.time * FRAMERATE:
    # Set the lookat point to the humanoid's center of mass.
    #camera.lookat = data.body('LH_SHANK').subtree_com
    #camera.lookat = data.qpos[0]

    renderer.update_scene(data, camera)
    pixels = renderer.render()
    frames.append(pixels.copy())

media.show_video(frames, fps=FRAMERATE)

'''
dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

for i in range(20):
  print(q_position[i])

ax[0].plot(timevals, q_position)
ax[0].set_title('q position')
ax[0].set_ylabel('position')

ax[1].plot(timevals, sensordata)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('sensor velocity')
_ = ax[1].set_title('sensor velocity')
'''