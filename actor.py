import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RandomActor:
    def __init__(self, env):
        self.env = env

    def get_action(self, states, device):
        assert len(states.shape) == 1, "can't work with batches"
        return self.env.action_space.sample()


class TD3_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = 512

        self.extract_features = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=self.hidden_size),
            nn.ReLU(),
        )
        self.forward_policy = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.action_dim),
            nn.Tanh(),
        )

    def get_action(self, states, device, std_noise=0.1):
        """
        Used to collect data by interacting with environment,
        so your have to add some noise to actions.
        input:
            states - numpy, (batch_size x features)
        output:
            actions - numpy, (batch_size x actions_dim)
        """
        # no gradient computation is required here since we will use this only for interaction
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            hidden_states = self.extract_features(states)
            policy = self.forward_policy(hidden_states)
            distribution = torch.distributions.normal.Normal(
                loc=torch.tensor([0.0]), scale=torch.tensor([std_noise])
            )
            noise = distribution.rsample(policy.size()).squeeze().to(device)
            noised_policy = policy + noise
            actions = np.clip(noised_policy.cpu().detach().numpy(), -1, 1)

            assert isinstance(
                actions, (list, np.ndarray)
            ), "convert actions to numpy to send into env"
            assert (
                actions.max() <= 1.0 and actions.min() >= -1
            ), "actions must be in the range [-1, 1]"
            return actions

    def get_best_action(self, states):
        """
        Will be used to optimize actor. Requires differentiable w.r.t. parameters actions.
        input:
            states - PyTorch tensor, (batch_size x features)
        output:
            actions - PyTorch tensor, (batch_size x actions_dim)
        """
        states = (
            torch.from_numpy(states).to(DEVICE).float()
            if not torch.is_tensor(states)
            else states
        )
        hidden_states = self.extract_features(states)
        actions = self.forward_policy(hidden_states)

        assert (
            actions.requires_grad
        ), "you must be able to compute gradients through actions"
        return actions

    def get_target_action(self, states, device, std_noise=0.2, clip_eta=0.5):
        """
        Will be used to create target for critic optimization.
        Returns actions with added "clipped noise".
        input:
            states - PyTorch tensor, (batch_size x features)
        output:
            actions - PyTorch tensor, (batch_size x actions_dim)
        """
        # no gradient computation is required here since we will use this only for interaction
        with torch.no_grad():
            # states = torch.from_numpy(states).to(DEVICE).float()
            states = states.float()
            hidden_states = self.extract_features(states)
            policy = self.forward_policy(hidden_states)
            # clipped_noise = torch.empty(self.action_dim).normal_(mean=0, std=std_noise)
            distribution = torch.distributions.normal.Normal(
                loc=torch.tensor([0.0]), scale=torch.tensor([std_noise])
            )
            sampled_noise = distribution.rsample(policy.size()).squeeze().to(device)
            clipped_noise = sampled_noise.clamp(-clip_eta, clip_eta).to(device)
            actions = policy + clipped_noise
            # actions can fly out of [-1, 1] range after added noise
            return actions.clamp(-1, 1)
