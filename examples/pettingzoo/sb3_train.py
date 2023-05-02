# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary to run Stable Baselines 3 agents on meltingpot substrates."""

import gymnasium
import stable_baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common import torch_layers
from stable_baselines3.common import vec_env
import supersuit as ss
import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

from examples.pettingzoo import utils
from meltingpot.python import substrate
from MADDPG import MADDPG



device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


# Use this with lambda wrapper returning observations only
class CustomCNN(torch_layers.BaseFeaturesExtractor):
  """Class describing a custom feature extractor."""

  def __init__(
      self,
      observation_space: gymnasium.spaces.Box,
      features_dim=128,
      num_frames=6,
      fcnet_hiddens=(1024, 128),
      flat_out = 36 * 7 * 7
  ):
    """Construct a custom CNN feature extractor.

    Args:
      observation_space: the observation space as a gymnasium.Space
      features_dim: Number of features extracted. This corresponds to the number
        of unit for the last layer.
      num_frames: The number of (consecutive) frames to feed into the network.
      fcnet_hiddens: Sizes of hidden layers.
    """
    super(CustomCNN, self).__init__(observation_space, features_dim)
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper

    self.conv = nn.Sequential(
        nn.Conv2d(
            num_frames * 3, num_frames * 3, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),  # 18 * 21 * 21 for arena; 18 * 9 * 9 for matrix
        nn.Conv2d(
            num_frames * 3, num_frames * 6, kernel_size=5, stride=2, padding=0),
        nn.ReLU(),  # 36 * 9 * 9 for arena; 36 * 3 * 3 for matrix
        nn.Conv2d(
            num_frames * 6, num_frames * 6, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),  # 36 * 7 * 7 for arena; 36 * 1 * 1 for matrix
        nn.Flatten(),
    )
    if (self._observation_space.shape[0] == 84 and self._observation_space.shape[1] == 84):     # arena
        flat_out = num_frames * 6 * 7 * 7
    elif (self._observation_space.shape[0] == 40 and self._observation_space.shape[1] == 40):   # repeated
        flat_out = num_frames * 6 * 1 * 1
    self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
    self.fc2 = nn.Linear(
        in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

  def forward(self, observations) -> torch.Tensor:
    # Convert to tensor, rescale to [0, 1], and convert from
    #   B x H x W x C to B x C x H x W
    observations = observations.permute(0, 3, 1, 2)
    features = self.conv(observations)
    features = F.relu(self.fc1(features))
    features = F.relu(self.fc2(features))
    return features


def main():
  # Config
  substrate_name = "bach_or_stravinsky_in_the_matrix__repeated"
  player_roles = substrate.get_config(substrate_name).default_player_roles
  env_config = {"substrate": substrate_name, "roles": player_roles}

  env = utils.parallel_env(render_mode="rgb_array", env_config=env_config)

  rollout_len = 1000
  total_timesteps = 2000000
  num_agents = env.max_num_agents

  # Training
  num_cpus = 1  # number of cpus
  num_envs = 1  # number of parallel multi-agent environments
  # number of frames to stack together; use >4 to avoid automatic
  # VecTransposeImage
  num_frames = 6
  # output layer of cnn extractor AND shared layer for policy and value
  # functions
  features_dim = 128
  fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
  ent_coef = 0.001  # entropy coefficient in loss
  batch_size = (rollout_len * num_envs // 2
               )  # This is from the rllib baseline implementation
  lr = 0.0001
  n_epochs = 30
  gae_lambda = 1.0
  gamma = 0.99
  target_kl = 0.01
  grad_clip = 40
  verbose = 3
  model_path = None  # Replace this with a saved model


  ## Parameters for DDPG
  episode_num = 30000
  episode_len = 25
  learn_interval = 100
  random_steps = 5e4

  buffer_capacity = 1000000
  batch_size = 1024
  gamma = 0.95
  tau = 0.02  
  actor_lr = 0.01
  critic_lr = 0.01
  

  env = utils.parallel_env(render_mode="rgb_array", env_config=env_config, max_cycles=rollout_len)
  print("1",type(env))

  print("1",type(env.observation_space('agent_0')))
  print("1",type(env.action_space('agent_0')))
  env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
  print("2",type(env))
  print("2",type(env.observation_space('agent_0')))
  print("2",type(env.action_space('agent_1')))

  new_env = env
  new_env.reset()
  dim_info = {}
  for agent_id in new_env.agents:
     dim_info[agent_id] = []
     print("3",type(new_env.observation_space(agent_id)))
     dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0:3])
     dim_info[agent_id].append(new_env.action_space(agent_id).n)


  env_dir = os.path.join('./results', substrate_name)
  if not os.path.exists(env_dir):
      os.makedirs(env_dir)
  total_files = len([file for file in os.listdir(env_dir)])
  result_dir = os.path.join(env_dir, f'{total_files + 1}')
  os.makedirs(result_dir)

  print("MAIN",dim_info)

  maddpg = MADDPG(dim_info= dim_info, capacity= buffer_capacity, batch_size=batch_size, actor_lr=actor_lr, critic_lr=critic_lr, res_dir = result_dir)

  step = 0 
  agent_num = env.num_agents
  episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
  for episode in range(episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            maddpg.add(obs, action, reward, next_obs, terminated | truncated)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= random_steps and step % learn_interval == 0:  # learn every few steps
                maddpg.learn(batch_size, gamma)
                maddpg.update_target(tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

  maddpg.save(episode_rewards)  # save model


  def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward   
  
  fig, ax = plt.subplots()
  x = range(1, episode_num + 1)
  for agent_id, reward in episode_rewards.items():
      ax.plot(x, reward, label=agent_id)
      ax.plot(x, get_running_reward(reward))
  ax.legend()
  ax.set_xlabel('episode')
  ax.set_ylabel('reward')
  title = f'training result of maddpg solve {substrate_name}'
  ax.set_title(title)
  plt.savefig(os.path.join(result_dir, title)) 
     
  # env = ss.pettingzoo_env_to_vec_env_v1(env)
  # print("3",type(env))
  # print("3",(env.par_env.observation_space('agent_0')))
  # print("3",(env.par_env.observation_space('agent_1')))
  # print("3",(env.par_env.action_space('agent_0')))
  # print("3",(env.par_env.action_space('agent_1')))

  # print("3", type(env.observation_space))
  # print("3",type(env.action_space))
#   env = ss.concat_vec_envs_v1(
#       env,
#       num_vec_envs=num_envs,
#       num_cpus=num_cpus,
#       base_class="stable_baselines3")
#   print("4",type(env))   
#   print("4",type(env.observation_space)) 
#   print("4",type(env.action_space))
#   env = vec_env.VecMonitor(env)
#   print("5",type(env))
#   print("5",type(env.observation_space))
#   print("5",type(env.action_space))
  # env = vec_env.VecTransposeImage(env, True)
  # print("6",type(env))
  # print("6",type(env.observation_space))
  # print("6",type(env.action_space))
#   env = vec_env.VecFrameStack(env, num_frames)
#   print("7",type(env))
#   print("7",type(env.observation_space))
#   print("7",type(env.action_space))

if __name__ == "__main__":
  main()
