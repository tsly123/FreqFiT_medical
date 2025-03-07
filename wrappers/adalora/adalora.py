import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper
from peft import AdaLoraConfig, get_peft_model


class AdaLoRAWrapper(BaseWrapper):
    def __init__(self, model, lora_targets, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = get_peft_model(model, AdaLoraConfig(target_modules=lora_targets))
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # MedMAE
        self.model.base_model.model.model.head = torch.nn.Linear(768, 2)
        


    def forward(self, x):
        return self.model(x)    # MedMAE