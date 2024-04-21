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
from distill import StudentModel
from stable_baselines3.common.callbacks import EvalCallback

MODEL_PATH = "pong"
POLICY_TYPE = "CnnPolicy"
ENV_NAME = "PongNoFrameskip-v4"
#Default frame stack used in original DQN paper and subsequent works
N_FRAMES_STACK = 4
NUM_ENVS = 50

def setup_env(n_envs, seed=0):
    env = make_atari_env(ENV_NAME, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=N_FRAMES_STACK)
    return env

def setup_model(n_envs, n_steps, device):
    env = setup_env(n_envs)
    model = PPO(POLICY_TYPE, env, verbose=1, n_steps=n_steps, device=device)
    return model, setup_env()

def train(model, save_path=MODEL_PATH, timesteps=int(2e6)):
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
            model = PPO.load(args.model_path, verbose=1, n_steps=args.n_steps, env = setup_env(), device=args.device)
    else:
        if args.evaluate:
            model = PPO.load(MODEL_PATH)
        else:
            model, _ = setup_model(args.n_envs, args.n_steps, device)

    if not args.evaluate: 
#        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
#                                     log_path="./logs/", eval_freq=500,
#                                     deterministic=True, render=False)
        train(model)

    # rendering seems to only work with 1 eval env
    if args.render:
        eval_env = make_atari_env(ENV_NAME, 1, env_kwargs={"render_mode":"human"})
    else:
        eval_env = make_atari_env(ENV_NAME, 10)
    
    eval_env = VecFrameStack(eval_env, N_FRAMES_STACK)

    mean_reward, std_reward = evaluate_policy(model, eval_env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent for Pong")
    parser.add_argument("--evaluate", action="store_true", help="Load and evaluate an existing model")
    parser.add_argument("--model_path", type=str, help="Path to the existing model")
    parser.add_argument("--torch_model", action="store_true", default=False, help="If model is regular torch model (distilled student model)")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--device", type=str, default="mps", help="Device, specify 'cpu' or 'gpu'")
    parser.add_argument("--n_steps", type=int, default=256, help="Number of episode steps before PPO weights update, we choose 256 because a losing episode is roughly 200 steps per point")
    args = parser.parse_args()

    main(args)




    
