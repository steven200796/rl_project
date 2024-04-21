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
ENV_NAME = "PongNoFrameskip-v4"

def setup_env():
    env = make_atari_env(ENV_NAME, n_envs=40, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env

def setup_model(device="mps"):
    env = setup_env()
    model = PPO("CnnPolicy", env, verbose=1, n_steps=64, device="mps")
    return model, setup_env()

def train(model, save_path=MODEL_PATH, timesteps=int(1e6)):
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(MODEL_PATH)

#Code modified from SB3 for local evaluation tuning
def evaluate_policy_local(model, env, render, n_eval_episodes=10):
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
        if args.torch_model:
            model = StudentModel.load(args.model_path, device=args.device)
        else:
            model = PPO.load(args.model_path, verbose=1, n_steps=64, env = setup_env(), device=args.device)
    else:
        if args.evaluate:
            model = PPO.load(MODEL_PATH)
        else:
            model = setup_model(device=args.device)

    if not args.evaluate: 
        train(model)

    eval_env = VecFrameStack(make_atari_env(ENV_NAME, 10), 4)#, env_kwargs={"render_mode":"human", "mode":0, "difficulty":0})
    print(model,eval_env)

    mean_reward, std_reward = evaluate_policy(model, eval_env)#, render=args.render)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent for Pong")
    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate an existing model")
    parser.add_argument("--model_path", type=str, help="Path to the existing model")
    parser.add_argument("--torch_model", action="store_true", default=False, help="If model is regular torch model (distilled student model)")
    parser.add_argument("--render", type=bool, default=False, help="If model is regular torch model")
    parser.add_argument("--device", type=str, default="mps", help="Device, specify 'cpu' or 'gpu'")
    args = parser.parse_args()

    main(args)




    
