import torch
import torch.nn as nn
def apply_rand_init(model):
    """
    re-init all the weight in the model with normal distribution
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            nn.init.normal_(param, mean=0, std=0.1)
