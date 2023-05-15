from torch import nn 

class SpuCoModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module, 
        representation_dim: int,
        num_classes: int
    ):
        super().__init__()
        self.backbone = backbone 
        self.classifier = nn.Linear(representation_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))