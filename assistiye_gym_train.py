from assistive_gym.learn import train, render_policy, evaluate_policy
import gym

env_name = 'BedBathingBaxter-v1'
algo = 'ppo'
policy_path = train(env_name, algo, timesteps_total=100000, save_dir='')

#env = gym.make('assistive_gym:'+env_name)
#env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
#render_policy(env, env_name, algo, policy_path, colab=True, seed=0)