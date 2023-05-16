from enum import Enum
from typing import Tuple
from spuco.models import MLP, LeNet, SpuCoModel

class SupportedModels(Enum):
    MLP = "mlp"
    #ConvNet = "convnet"
    #ResNet18 = "resnet18"
    #ResNet50 = "resnet50" 
    LeNet = "lenet"
    #AlexNet = "alexnet"
    #VGG11 = "vgg11"

def model_factory(arch: str, input_shape: Tuple[int, int, int], num_classes: int):

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

