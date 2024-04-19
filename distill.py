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

# This contains is modified from SB3
class StudentModel(BasePolicy):
    def __init__(
        self,
        obs_space,
        act_space,
        features_dim: int = 512,
    ) -> None:
        super(StudentModel, self).__init__(obs_space, act_space)
        n_input_channels = self.observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(obs_space.sample().transpose(2,0,1)[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(), nn.Linear(features_dim, act_space.n))

    def forward(self, obs): 
        return self.linear(self.cnn(obs))

    def _predict(self, obs, deterministic = False):
        logits = self.forward(obs)
        action_probs = torch.softmax(logits, dim=1)

        if deterministic:
            return torch.argmax(logits), action_probs
        else:
            return torch.multinomial(action_probs, num_samples=1).squeeze(1), action_probs



MODEL_PATH = "pong"
ENV_NAME = "ALE/Pong-v5"
ENV_NAME = "PongNoFrameskip-v4"

def distill(teacher_model, student_model, env, student_led, n_iter, criterion, optimizer):
#    student_model.train()
#    teacher_model.eval()
    eval_env = VecFrameStack(make_atari_env(ENV_NAME, 1), 4)

    #stepwise distillation, maybe batch / trajectory would do better?
    for i in range(n_iter):
        obs = env.reset()
        done = False
        ep_rew = 0
        while not done:
            obs = obs_as_tensor(obs.transpose(0,3,1,2), teacher_model.policy.device).float()
            old_obs = obs
            teacher_probs = teacher_model.policy.get_distribution(obs).distribution.probs
            _, student_probs = student_model._predict(obs, deterministic=True)

#            teacher_outputs = teacher_outputs / temperature
#            student_outputs = student_outputs / temperature 


            action_dist = student_probs if args.student_led else teacher_probs
            action = torch.argmax(action_dist, dim=1)
            obs, rew, done, _ = env.step(action)
            ep_rew += rew

            # Compute the loss
            loss = criterion(torch.log(student_probs), teacher_probs)
            #print("student:", student_probs, "teacher:", teacher_probs)
            #print(loss)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #with torch.no_grad():
                #print(student_model._predict(old_obs))
        print(i, ep_rew)
#        if i % 10:
#            evaluate_policy(student_model, env, render=False, n_eval_episodes=10)

def main(args):        
    env = make_atari_env(ENV_NAME, 1)
    env = VecFrameStack(env, n_stack = 4)

    teacher_model = PPO.load(args.teacher_path, device=args.device)
    if args.student_path:
        student_model = load_from_zip_file(args.student_path)
    else:
        student_model = StudentModel(env.observation_space, env.action_space).to(args.device)

    criterion = nn.KLDivLoss()#reduction='batchmean')
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001)

    distill(teacher_model, student_model, env, args.student_led, args.n, criterion, optimizer)

    save_to_zip_file(student_model, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a student agent for Pong")
    parser.add_argument("--teacher_path", required=True, type=str, help="Path to the teacher model")
    parser.add_argument("--student_path", type=str, help="Path to the student model")
    parser.add_argument("--save_path", default="student", type=str, help="Path to the student model save path")
    parser.add_argument("--student-led", type=bool, default=True, help="Model that generates the trajectories for distillation")
    parser.add_argument("--n", type=int, default=1000, help="Number of episodes to distill over")
    parser.add_argument("--device", type=str, default="mps") 

    args = parser.parse_args()

    main(args)
