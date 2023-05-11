import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):
        """
        Initializes the MLP network with three fully connected layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the MLP given an input tensor.

        :param x: Input tensor of size (batch_size, 28, 28).
        :type x: torch.Tensor

        :return: Output tensor of size (batch_size, 10) after passing through the MLP.
        :rtype: torch.Tensor
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
