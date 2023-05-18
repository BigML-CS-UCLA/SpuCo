from enum import Enum
from typing import Tuple
from spuco.models import MLP, LeNet, SpuCoModel

class SupportedModels(Enum):
    """
    Enum listing all supported models.
    """
    MLP = "mlp"
    LeNet = "lenet"

def model_factory(arch: str, input_shape: Tuple[int, int, int], num_classes: int):
    """
    Factory function to create a SpuCoModel based on the specified architecture.

    :param arch: The architecture name.
    :type arch: str
    :param input_shape: The shape of the input data in the format (channels, height, width).
    :type input_shape: Tuple[int, int, int]
    :param num_classes: The number of output classes.
    :type num_classes: int
    :return: A SpuCoModel instance.
    :rtype: SpuCoModel
    :raises NotImplementedError: If the specified architecture is not supported.
    """
    
    arch = SupportedModels(arch)
    channel = input_shape[0]
    image_size = (input_shape[1], input_shape[2])
    backbone = None
    representation_dim = -1 
    
    if arch == SupportedModels.MLP:
        backbone = MLP(channel * image_size[0] * image_size[1])
        representation_dim = backbone.representation_dim
    elif arch == SupportedModels.LeNet: 
        backbone = LeNet(channel=channel)
        representation_dim = backbone.representation_dim
    else:
        raise NotImplemented(f"Model {arch} not supported currently")
    return SpuCoModel(
        backbone=backbone, 
        representation_dim=representation_dim, 
        num_classes=num_classes
    )

