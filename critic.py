import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.extract_features = nn.Sequential(
            nn.Linear(
                in_features=state_dim + action_dim, out_features=self.hidden_size
            ),
            nn.ReLU(),
        )
        self.forward_q_values = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1),
        )

    def get_qvalues(self, states, actions):
        """
        input:
            states - tensor, (batch_size x features)
            actions - tensor, (batch_size x actions_dim)
        output:
            qvalues - tensor, critic estimation, (batch_size)
        """
        concatenated_tensor = torch.cat([states, actions], dim=-1)
        hidden_states = self.extract_features(concatenated_tensor)
        qvalues = self.forward_q_values(hidden_states)

        qvalues = torch.squeeze(qvalues)
        assert len(qvalues.shape) == 1 and qvalues.shape[0] == states.shape[0]

        return qvalues
