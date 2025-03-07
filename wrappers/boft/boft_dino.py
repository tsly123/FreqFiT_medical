import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper
from peft import BOFTConfig, get_peft_model


class BOFTWrapper_DINO(BaseWrapper):
    def __init__(self, model, lora_targets, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = get_peft_model(model, BOFTConfig(target_modules=lora_targets))

        for name, param in self.model.named_parameters():
            if "boft" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.model.base_model.model.model.head = torch.nn.Linear(768, 2)

    def forward(self, x):
        return self.model(x) 