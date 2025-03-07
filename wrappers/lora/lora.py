import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper
from peft import LoraConfig, get_peft_model


class LoRAWrapper(BaseWrapper):
    def __init__(self, model, lora_targets, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = get_peft_model(model, LoraConfig(target_modules=lora_targets))

        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
                
        # MedMAE
        self.model.base_model.model.model.head = torch.nn.Linear(768, 2)
        


    def forward(self, x):
        return self.model(x)    # MedMAE