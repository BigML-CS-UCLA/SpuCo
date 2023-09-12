import random as random

import numpy as np
import torch
from transformers import BertModel

from spuco.utils.random_seed import seed_randomness

# Acknowledgement to
# https://github.com/p-lambda/wilds/blob/main/examples/models/bert/bert.py


class Bert(BertModel):
    def __init__(self, config):
        """
        Initializes a BERT model.

        :param config: Configuration for the BERT model.
        :type config: transformers.configuration_bert.BertConfig
        """
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        """
        Forward pass of the BERT model.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1] # get pooled output
        return outputs