from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym

from stable_baselines3.common.evaluation import evaluate_policy

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multiprocessing training (num_env=4 => 4 processes)
env_name = 'PongNoFrameskip-v4'
env = make_atari_env(env_name, n_envs=6, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=25000, progress_bar=True)

obs = env.reset()

env = gym.make(env_name, render_mode="human")

mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
