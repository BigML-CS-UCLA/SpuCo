import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from spuco.group_inference import BaseGroupInference
from spuco.utils.random_seed import seed_randomness


class CorrectNContrastInference(BaseGroupInference):
    """
    Correct-n-Contrast Inference: https://proceedings.mlr.press/v162/zhang22z.html
    """
    def __init__(
        self,
        logits: torch.Tensor,
        verbose: bool = False
    ):
        """
        Initializes the CorrectNContrastInference object.

        :param logits: The output of the network.
        :type logits: torch.Tensor
        :param verbose: Whether to print training progress. Defaults to False.
        :type verbose: bool, optional
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        super().__init__()
        self.logits = logits
        self.verbose = verbose

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Performs Correct-n-Contrast inference to infer group partitions.

        :return: The group partition based on Correct-n-Contrast inference.
        :rtype: Dict[Tuple[int, int], List[int]]
        """
        
        spurious = torch.argmax(self.logits, dim=-1).cpu().tolist()

        # CNC only wants to group based on spurious attribute, but for consistency in API
        # we need a tuple as key, hence group_label set to (0, spurious) 
        group_partition = {}
        for i in range(len(spurious)):
            group_label = (0, self.spurious[i])
            if group_label not in group_partition:
                group_partition[group_label] = []
            group_partition[group_label].append(i)
            
        return group_partition

