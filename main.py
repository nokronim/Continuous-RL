import sys, os

import gymnasium as gym
import numpy as np


env_name = 'HalfCheetah-v4'

env = gym.make(env_name, render_mode="rgb_array")

# we want to look inside
env.reset()

# examples of states and actions
print("observation space: ", env.observation_space,
      "\nobservations:", env.reset()[0])
print("action space: ", env.action_space,
      "\naction_sample: ", env.action_space.sample())

import matplotlib.pyplot as plt

class RandomActor():
    def get_action(self, states):
        assert len(states.shape) == 1, "can't work with batches"
        return env.action_space.sample()

s, _ = env.reset()
rewards_per_step = []
actor = RandomActor()

for i in range(10000):
    a = actor.get_action(s)
    s, r, terminated, truncated, _ = env.step(a)

    rewards_per_step.append(r)

    if terminated or truncated:
        s, _ = env.reset()
        print("done: ", i)

env.close()


from logger import TensorboardSummaries as Summaries

env = gym.make(env_name, render_mode="rgb_array")
env = Summaries(env, "MyFirstWalkingAnt")

state_dim = env.observation_space.shape[0]  # dimension of state space (27 numbers)
action_dim = env.action_space.shape[0]      # dimension of action space (8 numbers)

import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()        

        self.hidden_size = 512
        
        self.extract_features = nn.Sequential(
            nn.Linear(in_features=state_dim + action_dim, out_features=self.hidden_size),
            nn.ReLU()
        )
        self.forward_q_values = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1)
        )
        

    def get_qvalues(self, states, actions):
        '''
        input:
            states - tensor, (batch_size x features)
            actions - tensor, (batch_size x actions_dim)
        output:
            qvalues - tensor, critic estimation, (batch_size)
        '''
        concatenated_tensor = torch.cat([states, actions], dim=-1)
        hidden_states = self.extract_features(concatenated_tensor)
        qvalues = self.forward_q_values(hidden_states)

        qvalues = torch.squeeze(qvalues)
        assert len(qvalues.shape) == 1 and qvalues.shape[0] == states.shape[0]
        
        return qvalues


# template for TD3; template for SAC is below
class TD3_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()        

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = 512
        
        self.extract_features = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=self.hidden_size),
            nn.ReLU()
        )
        self.forward_policy = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.action_dim),
            nn.Tanh()
        )

    def get_action(self, states, std_noise=0.1):
        '''
        Used to collect data by interacting with environment,
        so your have to add some noise to actions.
        input:
            states - numpy, (batch_size x features)
        output:
            actions - numpy, (batch_size x actions_dim)
        '''
        # no gradient computation is required here since we will use this only for interaction
        with torch.no_grad():
            states = torch.from_numpy(states).to(DEVICE).float()
            hidden_states = self.extract_features(states)
            policy = self.forward_policy(hidden_states)
            distribution = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), 
                                                              scale=torch.tensor([std_noise]))
            noise = distribution.rsample(policy.size()).squeeze().to(DEVICE)
            noised_policy = policy + noise
            actions = np.clip(noised_policy.cpu().detach().numpy(), -1, 1)
                                                  
            assert isinstance(actions, (list,np.ndarray)), "convert actions to numpy to send into env"
            assert actions.max() <= 1. and actions.min() >= -1, "actions must be in the range [-1, 1]"
            return actions
    
    def get_best_action(self, states):
        '''
        Will be used to optimize actor. Requires differentiable w.r.t. parameters actions.
        input:
            states - PyTorch tensor, (batch_size x features)
        output:
            actions - PyTorch tensor, (batch_size x actions_dim)
        '''
        states = torch.from_numpy(states).to(DEVICE).float() if not torch.is_tensor(states) else states
        hidden_states = self.extract_features(states)
        actions = self.forward_policy(hidden_states)
        
        assert actions.requires_grad, "you must be able to compute gradients through actions"
        return actions
    
    def get_target_action(self, states, std_noise=0.2, clip_eta=0.5):
        '''
        Will be used to create target for critic optimization.
        Returns actions with added "clipped noise".
        input:
            states - PyTorch tensor, (batch_size x features)
        output:
            actions - PyTorch tensor, (batch_size x actions_dim)
        '''
        # no gradient computation is required here since we will use this only for interaction
        with torch.no_grad():
            # states = torch.from_numpy(states).to(DEVICE).float()
            states = states.float()
            hidden_states = self.extract_features(states)
            policy = self.forward_policy(hidden_states)
            # clipped_noise = torch.empty(self.action_dim).normal_(mean=0, std=std_noise)
            distribution = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), 
                                                              scale=torch.tensor([std_noise]))
            sampled_noise = distribution.rsample(policy.size()).squeeze().to(DEVICE)
            clipped_noise = sampled_noise.clamp(-clip_eta, clip_eta).to(DEVICE)
            actions = policy + clipped_noise
            # actions can fly out of [-1, 1] range after added noise
            return actions.clamp(-1, 1)


import random


class ReplayBuffer():
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''        
        data = (obs_t, action, reward, obs_tp1, done)
        storage = self._storage
        maxsize = self._maxsize
        
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        storage = self._storage
        # randomly generate batch_size integers
        # to be used as indexes of samples
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

        # collect <s,a,r,s',done> for each index
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        # <states>, <actions>, <rewards>, <next_states>, <is_done>
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        
        # select action using policy with exploration
        a = agent.get_action(s)
        
        ns, r, terminated, truncated, _ = env.step(a)
        
        exp_replay.add(s, a, r, ns, terminated)
        
        s = env.reset()[0] if terminated or truncated else ns
        
        sum_rewards += r        

    return sum_rewards, s


gamma=0.99                    # discount factor
max_buffer_size = 10**5       # size of experience replay
start_timesteps = 5000        # size of experience replay when start training
timesteps_per_epoch=1         # steps in environment per step of network updates
batch_size=4096               # batch size for all optimizations
max_grad_norm=10              # max grad norm for all optimizations
tau=0.005                     # speed of updating target networks
policy_update_freq=2          # frequency of actor update; vanilla choice is 2 for TD3 or 1 for SAC
alpha=0.1                     # temperature for SAC

# iterations passed
n_iterations = 1000


exp_replay = ReplayBuffer(max_buffer_size)


# models to train
actor = TD3_Actor(state_dim, action_dim).to(DEVICE)
critic1 = Critic(state_dim, action_dim).to(DEVICE)
critic2 = Critic(state_dim, action_dim).to(DEVICE)

# target networks: slow-updated copies of actor and two critics
target_critic1 = Critic(state_dim, action_dim).to(DEVICE)
target_critic2 = Critic(state_dim, action_dim).to(DEVICE)
target_actor = TD3_Actor(state_dim, action_dim).to(DEVICE)  # comment this line if you chose SAC

# initialize them as copies of original models
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())
target_actor.load_state_dict(actor.state_dict())            # comment this line if you chose SAC 


def update_target_networks(model, target_model):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# optimizers: for every model we have
opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=3e-4)
opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=3e-4)


# just to avoid writing this code three times
def optimize(name, model, optimizer, loss):
    '''
    Makes one step of SGD optimization, clips norm with max_grad_norm and 
    logs everything into tensorboard
    '''
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # logging
    env.writer.add_scalar(name, loss.item(), n_iterations)
    env.writer.add_scalar(name + "_grad_norm", grad_norm.item(), n_iterations)


def compute_critic_target(rewards, next_states, is_done):
    '''
    Important: use target networks for this method! Do not use "fresh" models except fresh policy in SAC!
    input:
        rewards - PyTorch tensor, (batch_size)
        next_states - PyTorch tensor, (batch_size x features)
        is_done - PyTorch tensor, (batch_size)
    output:
        critic target - PyTorch tensor, (batch_size)
    '''
    gamma = 0.997
    with torch.no_grad():
        critic_target = rewards + gamma*(1 - is_done)*torch.min(target_critic1.get_qvalues(next_states, target_actor.get_target_action(next_states)), 
                                                                target_critic2.get_qvalues(next_states, target_actor.get_target_action(next_states)))
    
    assert not critic_target.requires_grad, "target must not require grad."
    assert len(critic_target.shape) == 1, "dangerous extra dimension in target?"

    return critic_target


def compute_actor_loss(states):
    '''
    Returns actor loss on batch of states
    input:
        states - PyTorch tensor, (batch_size x features)
    output:
        actor loss - PyTorch tensor, (batch_size)
    '''
    # make sure you have gradients w.r.t. actor parameters
    actions = actor.get_best_action(states)
    
    assert actions.requires_grad, "actions must be differentiable with respect to policy parameters"
    
    # compute actor loss
    # TD3
    actor_loss = -target_critic1.get_qvalues(states, actions)
    return actor_loss

seed = 42
np.random.seed(seed)
# env.unwrapped.seed(seed)
torch.manual_seed(seed)

from tqdm import trange

interaction_state, _ = env.reset(seed=seed)
random_actor = RandomActor()

iters = 100

for n_iterations in trange(0, iters, timesteps_per_epoch):
    # if experience replay is small yet, no training happens
    # we also collect data using random policy to collect more diverse starting data
    if len(exp_replay) < start_timesteps:
        _, interaction_state = play_and_record(interaction_state, random_actor, env, exp_replay, timesteps_per_epoch)
        continue
        
    # perform a step in environment and store it in experience replay
    _, interaction_state = play_and_record(interaction_state, actor, env, exp_replay, timesteps_per_epoch)
        
    # sample a batch from experience replay
    states, actions, rewards, next_states, is_done = exp_replay.sample(batch_size)
    
    # move everything to PyTorch tensors
    states = torch.tensor(states, device=DEVICE, dtype=torch.float)
    actions = torch.tensor(actions, device=DEVICE, dtype=torch.float)
    rewards = torch.tensor(rewards, device=DEVICE, dtype=torch.float)
    next_states = torch.tensor(next_states, device=DEVICE, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=DEVICE,
        dtype=torch.float
    )
    
    # losses
    critic1_loss = (critic1.get_qvalues(states, actions) - compute_critic_target(rewards, next_states, is_done)) ** 2
    optimize("critic1", critic1, opt_critic1, critic1_loss)

    critic2_loss = (critic2.get_qvalues(states, actions) - compute_critic_target(rewards, next_states, is_done)) ** 2
    optimize("critic2", critic2, opt_critic2, critic2_loss)

    # actor update is less frequent in TD3
    if n_iterations % policy_update_freq == 0:
        actor_loss = compute_actor_loss(states)
        optimize("actor", actor, opt_actor, actor_loss)

        # update target networks
        update_target_networks(critic1, target_critic1)
        update_target_networks(critic2, target_critic2)
        update_target_networks(actor, target_actor)                     # comment this line if you chose SAC


def evaluate(env, actor, n_games=1, t_max=1000):
    '''
    Plays n_games and returns rewards and rendered games
    '''
    rewards = []

    for _ in range(n_games):
        s, _ = env.reset()

        R = 0
        for _ in range(t_max):
            # select action for final evaluation of your policy
            action = actor.get_best_action(s)

            assert (action.max() <= 1).all() and  (action.min() >= -1).all()

            s, r, terminated, truncated, _ = env.step(action.cpu().detach().numpy())

            R += r

            if terminated or truncated:
                break

        rewards.append(R)
    return np.array(rewards)


# evaluation will take some time!
sessions = evaluate(env, actor, n_games=1)
score = sessions.mean()
print(f"Your score: {score}")

# assert score >= 1000, "Needs more training?"
print("Well done!")


# from gymnasium.wrappers import RecordVideo

# # let's hope this will work
# # don't forget to pray
# with gym.make(env_name, render_mode="rgb_array") as env, RecordVideo(
#     env=env, video_folder="./videos"
# ) as env_monitor:
#     # note that t_max is 300, so collected reward will be smaller than 1000
#     evaluate(env_monitor, actor, n_games=1, t_max=300)


# # Show video. This may not work in some setups. If it doesn't
# # work for you, you can download the videos and view them locally.

# from pathlib import Path
# from base64 import b64encode
# from IPython.display import HTML
# import sys

# video_paths = sorted([s for s in Path('videos').iterdir() if s.suffix == '.mp4'])
# video_path = video_paths[0]  # You can also try other indices


# data_url = str(video_path)

# HTML("""
# <video width="640" height="480" controls>
#   <source src="{}" type="video/mp4">
# </video>
# """.format(data_url))
