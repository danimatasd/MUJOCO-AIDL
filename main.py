#sudo apt install ffmpeg
#git clone https://github.com/deepmind/mujoco
#env MUJOCO_GL=egl

#@title Other imports and helper functions
import mujoco
import numpy as np



from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
 
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


model2 = mujoco.MjModel.from_xml_path("mujoco_menagerie/anybotics_anymal_c/scene.xml")
data2 = mujoco.MjData(model2)
renderer2=mujoco.Renderer(model2)

'''renderer2.update_scene(data2)
print("Anymal C")
media.show_image(renderer2.render())'''

print(f"Timestep: {model2.opt.timestep}")

DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model2, camera)
camera.distance = 3

timevals = []
angular_velocity = []
stem_height = []
frames = []

# Simulate and save data
mujoco.mj_resetDataKeyframe(model2, data2, 0)

while data2.time < DURATION:
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
  data2.ctrl = (lfkrot+rfkrot+lfsrot+rfsrot)*np.sin(data2.time)*(-0.2)+(lbkrot+rbkrot+lbsrot+rbsrot)*np.cos(data2.time*2)*(-0.2)
  
  # Step the simulation.
  mujoco.mj_step(model2, data2)
  
  timevals.append(data2.time)
  angular_velocity.append(data2.actuator_velocity)
  stem_height.append(data2.body('LF_HIP').subtree_com)

  # Render and save frames.
  if len(frames) < data2.time * FRAMERATE:
    # Set the lookat point to the humanoid's center of mass.
    camera.lookat = data2.body('LH_SHANK').subtree_com

    renderer2.update_scene(data2, camera)
    pixels = renderer2.render()
    frames.append(pixels.copy())

media.show_video(frames, fps=FRAMERATE)

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular position')
ax[0].set_ylabel('radians')

ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters')
_ = ax[1].set_title('stem height')



"""
with open('mujoco/model/humanoid/humanoid.xml', 'r') as f:
  xml = f.read()
  
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
print("Humanoid")
media.show_image(renderer.render())


for key in range(model.nkey):
  mujoco.mj_resetDataKeyframe(model, data, key)
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)
  print("Humanoid Positions")
  media.show_image(renderer.render())
  
DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Initialize to the standing-on-one-leg pose.
mujoco.mj_resetDataKeyframe(model, data, 1)

frames = []
while data.time < DURATION:
  # Step the simulation.
  mujoco.mj_step(model, data)

  # Render and save frames.
  if len(frames) < data.time * FRAMERATE:
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels.copy())

# Display video.
print("Humanoid Video")
media.show_video(frames, fps=FRAMERATE)

DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2

mujoco.mj_resetDataKeyframe(model, data, 1)

frames = []
while data.time < DURATION:
  # Set control vector.
  data.ctrl = np.random.randn(model.nu)

  # Step the simulation.
  mujoco.mj_step(model, data)

  # Render and save frames.
  if len(frames) < data.time * FRAMERATE:
    # Set the lookat point to the humanoid's center of mass.
    camera.lookat = data.body('torso').subtree_com

    renderer.update_scene(data, camera)
    pixels = renderer.render()
    frames.append(pixels.copy())

media.show_video(frames, fps=FRAMERATE)
"""