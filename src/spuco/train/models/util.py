from enum import Enum

from train_factory.resnet import ResNet10, ResNet18, ResNet34, ResNet50
from train_factory.mlp import MLP
from train_factory.lenet import LeNet
from train_factory.alexnet import AlexNet
from train_factory.vgg import VGG11 
from train_factory.convnet import ConvNet

class SupportedModels(Enum):
    MLP = "mlp"
    ConvNet = "convnet"
    ResNet10 = "resnet10"
    ResNet18 = "resnet18"
    ResNet34 = "resnet34"
    ResNet50 = "resnet50" 
    LeNet = "lenet"
    AlexNet = "alexnet"
    VGG11 = "vgg11"

def get_net(arch: str, stem, input_shape):

    arch = SupportedModels(arch)
    channel = input_shape[0]
    image_size = (input_shape[1], input_shape[2])

    if arch == SupportedModels.ResNet10:
        net = ResNet10(stem=stem)
    elif arch == SupportedModels.ResNet18:
        net = ResNet18(stem=stem)
    elif arch == SupportedModels.ResNet34:
        net = ResNet34(stem=stem)
    elif arch == SupportedModels.ResNet50:
        net = ResNet50(stem=stem)
    elif arch == SupportedModels.MLP:
        net = MLP(channel=channel)
    elif arch == SupportedModels.ConvNet:
        net = ConvNet(
            channel=channel,
            net_width=128,
            net_depth=3, 
            net_act='relu', 
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=image_size)
    elif arch == SupportedModels.LeNet:
        net = LeNet(channel=channel)
    elif arch == SupportedModels.AlexNet:
        net = AlexNet(channel=channel)
    elif arch == SupportedModels.VGG11:
        net = VGG11(channel=channel)
    else:
        raise ValueError("Unsupported archictecture specified")
    return net