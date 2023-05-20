import random as random

import numpy as np
import torch
from transformers import DistilBertModel

from spuco.utils.random_seed import seed_randomness

# Acknowledgement to
# https://github.com/p-lambda/wilds/blob/main/examples/models/bert/distilbert.py


class DistilBert(DistilBertModel):
    def __init__(self, config):
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output