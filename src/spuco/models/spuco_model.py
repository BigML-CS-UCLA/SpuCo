from torch import nn 

class SpuCoModel(nn.Module):
    """
    Wrapper module to allow for methods that use penultimate layer embeddings
    to easily access this via *backbone*
    """
    def __init__(
        self,
        backbone: nn.Module, 
        representation_dim: int,
        num_classes: int
    ):
        """
        Initializes a SpuCoModel 

        :param backbone: The backbone network.
        :type backbone: torch.nn.Module
        :param representation_dim: The dimensionality of the penultimate layer embeddings.
        :type representation_dim: int
        :param num_classes: The number of output classes.
        :type num_classes: int
        """
        super().__init__()
        self.backbone = backbone 
        self.classifier = nn.Linear(representation_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the SpuCoModel.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.classifier(self.backbone(x))