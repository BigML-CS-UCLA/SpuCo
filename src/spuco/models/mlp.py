import torch.nn as nn
 
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = 256
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, self.representation_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x
