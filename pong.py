import gymnasium as gym
import numpy as np
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_atari_env
from distill import StudentModel
from stable_baselines3.common.callbacks import EvalCallback

import logging
import os
import sys

POLICY_TYPE = "CnnPolicy"
ENV_NAME = "PongNoFrameskip-v4"
#Default frame stack used in original DQN paper and subsequent works
N_FRAMES_STACK = 4
TB_LOG = "./pong_run/"
# one game is 4k frames at minimum, default 100 window is too much
STATS_WINDOW_SIZE = 100

def transform_env_no_transpose(env):
    return VecFrameStack(env, N_FRAMES_STACK)

def transform_env(env):
    return VecTransposeImage(transform_env_no_transpose(env))

def setup_env(n_envs):
    env = make_atari_env(ENV_NAME, n_envs=n_envs)
    env = transform_env(env)
    return env

def setup_model(n_envs, n_steps, device):
    env = setup_env(n_envs)
    model = PPO(POLICY_TYPE, env, verbose=1, n_steps=n_steps, device=device, tensorboard_log=TB_LOG, stats_window_size=STATS_WINDOW_SIZE)
    return model

def train(model, timesteps, save_path, eval_callback=None):
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback, tb_log_name="expert_run")
    model.save(save_path)

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
            model = PPO.load(args.model_path, verbose=1, n_steps=args.n_steps, env = setup_env(args.n_envs), device=args.device)
    else:
        if args.eval_only:
            model = PPO.load(args.save_path)
        else:
            model = setup_model(args.n_envs, args.n_steps, args.device)

    eval_env = transform_env(make_atari_env(ENV_NAME, args.n_eval_episodes))
    if not args.eval_only: 
        eval_callback = EvalCallback(eval_env, best_model_save_path="./pong_run/logs/",
                                     log_path="./pong_run/logs/", n_eval_episodes=args.n_eval_episodes, eval_freq = max(args.eval_freq // args.n_envs, 1),
                                     deterministic=False, render=False, verbose=1)
        train(model, args.timesteps, args.save_path, eval_callback=eval_callback)

    # rendering seems to only work with 1 eval env
    if args.render:
        eval_env = transform_env_no_transpose(make_atari_env(ENV_NAME, 1, env_kwargs={"render_mode":"human"}))
    else:
        # Env has to be recreated because evaluate policy wants a non-transposed environment
        eval_env = transform_env_no_transpose(make_atari_env(ENV_NAME, 10))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    logging.info(f"Mean reward: {mean_reward} +/- {std_reward} (over 100 episodes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent for Pong")
    parser.add_argument("--eval_only", action="store_true", help="Load and evaluate an existing model")
    parser.add_argument("--model_path", type=str, help="Path to the existing model")
    parser.add_argument("--torch_model", action="store_true", default=False, help="If model is regular torch model (distilled student model)")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--device", type=str, default="mps", help="Device, specify 'cpu' or 'gpu'")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of episode steps before PPO weights update, we choose 256 because a losing episode is roughly 200 steps per point")
    parser.add_argument("--n_envs", type=int, default=50, help="Number of environments in vecenv")
    parser.add_argument("--eval_freq", type=int, default=5000, help="How frequently to evaluate the model (number of steps)")
    parser.add_argument("--n-eval_episodes", type=int, default=10, help="How many evaluation episodes")
    parser.add_argument("--timesteps", type=int, default=2e6, help="Number of training timesteps")
    parser.add_argument("--save_path", type=str, default=f"pong_run/pong_expert", help="Path to save trained model")

    args = parser.parse_args()

    os.makedirs("pong_run/", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format("pong_run", "pong")),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")

    main(args) 
