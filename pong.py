import gymnasium as gym
import numpy as np
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_atari_env
import torch
from distill import StudentModel

MODEL_PATH = "pong"
ENV_NAME = "ALE/Pong-v5"
ENV_NAME = "PongNoFrameskip-v4"
NUM_CPUS = 10

def setup_env():
    env = make_atari_env(ENV_NAME, n_envs=40, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env

def setup(BUFFER_SIZE=int(1e4)):
    env = setup_env()
    model = PPO("CnnPolicy", env, verbose=1, n_steps=64, device="mps")#, buffer_size=BUFFER_SIZE)
    return model, setup_env()

def train(model, save_path=MODEL_PATH, timesteps=int(1e6)):
    # Train the agent
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save the model
    model.save(MODEL_PATH)

def evaluate_policy_local(model, env, render, n_eval_episodes=10):
    """
    Evaluate a RL agent policy.

    :param model: The RL agent
    :param env: The Gym environment
    :param n_eval_episodes: Number of episodes to evaluate the agent
    :return: Mean and standard deviation of rewards obtained
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        episode_rewards.append(episode_reward)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

def main(args):
    if args.model_path:
#        env = setup_env()
        if args.torch_model:
            model = torch.load(args.model_path)
            model.to("mps")
        else:
            model = PPO.load(args.model_path, verbose=1, n_steps=64, env = setup_env(), device="mps")
#        model.set_env(env)
    else:
        if args.evaluate:
            model = PPO.load(MODEL_PATH)
        else:
            model = setup()

    if not args.evaluate: 
        train(model)
        args.render = True

    env = make_atari_env(ENV_NAME, 1, env_kwargs={"render_mode":"human", "mode":0, "difficulty":0})
    env = VecFrameStack(env, n_stack = 4)

    mean_reward, std_reward = evaluate_policy(model, env, render=args.render, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent for Pong")
    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate an existing model")
    parser.add_argument("--model_path", type=str, help="Path to the existing model")
    parser.add_argument("--torch_model", action="store_true", default=False, help="If model is regular torch model")
    parser.add_argument("--render", type=bool, default=False, help="If model is regular torch model")
    args = parser.parse_args()

    main(args)




    
