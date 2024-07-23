import torch
from torch import nn


class DQN(nn.Module):
    """
    Deep Q-learning Neural Network model.
    """

    def __init__(self, in_states: int, out_actions: int):
        super().__init__()
        # Constants
        self.L1_NODES = 48

        # Activation functions
        self.relu = nn.ReLU()

        # Layers
        self.block_1 = nn.Linear(in_features=in_states, out_features=self.L1_NODES)
        self.block_2 = nn.Linear(in_features=self.L1_NODES, out_features=out_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.relu(x)
        x = self.block_2(x)
        return x
