import gym
import numpy as np
import assistive_gym

env = gym.make('ValidationPR2-v1')

# Make a feeding assistance environment with the Jaco robot.
env.set_seed(200)
# Setup a camera in the environment to capture images (for rendering)
#env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)

# Reset the environment
observation = env.reset()
#frames = []
#done = False
#while not done:
    # Step the simulation forward. Have the robot take a random action.
   # observation, reward, done, info = env.step(env.action_space.sample())
   # # Capture (render) an image from the camera
   # img, depth = env.get_camera_image_depth()
  #  frames.append(img)
#env#.disconnect()

# Compile the camera frames into an animated png (apng)
#write_apng('output.png', frames, delay=100)