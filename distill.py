import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.policies import obs_as_tensor

import collections
import logging
import sys
import os

MODEL_PATH = "pong"
ENV_NAME = "PongNoFrameskip-v4"

# This contains code that's modified from SB3
class StudentModel(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        student_model_type,
        normalize_images=False,
        features_dim: int = 512,
    ) -> None:
        super(StudentModel, self).__init__(observation_space, action_space)
        n_input_channels = self.observation_space.shape[2]

        assert student_model_type in ("BCBL", "SCBL", "BCSL", "SCSL")

        if student_model_type == "BCBL" or student_model_type == "BCSL":
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample().transpose(2,0,1)[None]).float()).shape[1]

        if student_model_type == "BCBL" or student_model_type == "SCBL":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(), nn.Linear(features_dim, action_space.n))
        else:
            self.linear = nn.Linear(n_flatten, action_space.n)

    def forward(self, obs): 
        return self.linear(self.cnn(obs))

    def _predict(self, obs, deterministic = False, probs_only=False):
        # Channel first hack to integrate with sb3, we can't get distributions without calling this directly but it's also needed to integrate with sb3 evaluations
        logits = self.forward(obs.permute(0,3,1,2).float())
        action_probs = torch.softmax(logits, dim=1)

        if probs_only:
            return action_probs

        if deterministic:
            return torch.argmax(logits, dim=1)
        else:
            return torch.multinomial(action_probs, num_samples=1).squeeze(1)


def distill(teacher_model, student_model, env, student_led, n_iter, criterion, optimizer):
#    student_model.train()
#    teacher_model.eval()
    eval_env = VecFrameStack(make_atari_env(ENV_NAME, 10), 4)
    #mean_reward, std_reward = evaluate_policy(student_model, eval_env)
    #print(f"Mean reward: {mean_reward} +/- {std_reward}")
    obs = env.reset()

    npz_timesteps = []
    npz_results = []
    npz_ep_lengths = []

    #stepwise distillation, maybe batch / trajectory would do better?
    #future resets are automatically called in vecenv
    games_played = 0
    scores = collections.deque(maxlen=20)
    ep_rew = 0
    for i in range(int(n_iter)):
        done = False
#        while not done:
        obs = obs_as_tensor(obs, teacher_model.policy.device)#.float()
        teacher_probs = teacher_model.policy.get_distribution(obs.permute(0,3,1,2)).distribution.probs
        student_probs = student_model._predict(obs, probs_only=True)

#            teacher_outputs = teacher_outputs / temperature
#            student_outputs = student_outputs / temperature

        action_dist = student_probs if student_led else teacher_probs
        action = torch.argmax(action_dist, dim=1)
        obs, rew, done, _ = env.step(action)

        games_played += np.sum(done)

        ep_rew += rew
        scores.extendleft(ep_rew[done])
        ep_rew[done] = 0

        loss = criterion(torch.log(student_probs), teacher_probs)
        #print("student:", student_probs, "teacher:", teacher_probs)
        #print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i)
        if i % 500 == 0:
            rewards, ep_lengths = evaluate_policy(student_model, eval_env, n_eval_episodes=30, return_episode_rewards=True)
            rewards = np.array(rewards)
            ep_lengths = np.array(ep_lengths)
            logging.info(f"{i} Games Played: {games_played} Running average: {0 if len(scores) == 0 else sum(scores)/len(scores)} Mean reward: {rewards.mean()} +/- {rewards.std()} (over 30 episodes)")
            npz_timesteps.append(50 * i)
            npz_results.append(rewards)
            npz_ep_lengths.append(ep_lengths)
    
    mean_reward, std_reward = evaluate_policy(student_model, eval_env, n_eval_episodes=100)
    logging.info(f"{i} Games Played: {games_played} Running average: {sum(scores)/len(scores)} Mean reward: {mean_reward} +/- {std_reward} (over 100 episodes)")

    npz_timesteps = np.array(npz_timesteps)
    npz_results = np.array(npz_results)
    npz_ep_lengths = np.array(npz_ep_lengths)
    np.savez("distill_run/evaluations.npz", timesteps=npz_timesteps, results=npz_results, ep_lengths=npz_ep_lengths)

def main(args):
    env = make_atari_env(ENV_NAME, 50)
    env = VecFrameStack(env, n_stack = 4)

    teacher_model = PPO.load(args.teacher_path, device=args.device)
    if args.student_path:
#        student_model = torch.load(args.student_path)
        student_model = StudentModel.load(args.student_path, device=args.device)
    else:
        student_model = StudentModel(env.observation_space, env.action_space, args.student_model_type).to(args.device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001)

    distill(teacher_model, student_model, env, not args.teacher_led, args.n, criterion, optimizer)

#    torch.save(student_model, args.save_path)
    student_model.save(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a student agent for Pong")
    parser.add_argument("--teacher_path", required=True, type=str, help="Path to the teacher model")
    parser.add_argument("--student_path", type=str, help="Path to the student model")
    parser.add_argument("--save_path", default="distill_run/student.pt", type=str, help="Path to the student model save path")
    parser.add_argument("--teacher_led", action="store_true", help="Model that generates the trajectories for distillation")
    parser.add_argument("--n", type=int, default=20001, help="Number of episodes to distill over")
    parser.add_argument("--device", type=str, default="mps")
    # BCSL = "Big Convolutional (layer), Small Linear (layer)", etc.
    parser.add_argument("--student_model_type", type=str, default="BCBL", help="Which type of student model to use")

    args = parser.parse_args()

    os.makedirs("distill_run/", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format("distill_run", "distill")),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")

    main(args)
