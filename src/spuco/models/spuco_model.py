import random

import numpy as np
import torch
from torch import nn

from spuco.utils.random_seed import seed_randomness


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
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
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
    
    def get_penultimate_layer(self, x):
        """
        Returns the penultimate layer embeddings for the input tensor.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Penultimate layer embeddings.
        :rtype: torch.Tensor
        """
        return self.backbone(x)


    def get_gradcam_mask(self, x):
        """
        Computes the GradCAM mask for the input tensor.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: GradCAM mask.
        :rtype: torch.Tensor
        """
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        
        # Get penultimate layer embeddings
        penultimate_layer = self.get_penultimate_layer(x)
        
        # Compute gradients of the output w.r.t. the penultimate layer embeddings
        self.backbone.zero_grad()
        self.classifier.zero_grad()
        penultimate_layer.retain_grad()
        output = self.forward(x)
        
        # Compute GradCAM mask for each class
        masks = []
        for i in range(output.shape[1]):
            self.zero_grad()
            self.classifier.zero_grad()
            output[:, i].backward(retain_graph=True)
            mask = torch.mean(penultimate_layer.grad, dim=1)
            mask = torch.relu(mask)
            mask = mask / (torch.max(mask) + 1e-8)
            masks.append(mask)
        return torch.stack(masks, dim=1)
    
        
    


    

    
        

    
