import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from wrappers.base import BaseWrapper
import math
import timm

from torch import Tensor
from torch.nn.init import trunc_normal_

from wrappers.utils_models import vit_base, DinoVisionTransformer
from peft import BOFTConfig, get_peft_model
from wrappers.freqfit import GlobalFilter2D, init_ssf_scale_shift, ssf_ada

class VisionTransformer(DinoVisionTransformer):
    def __init__(self, freqfit_config=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.freqfit_config = freqfit_config

        if self.freqfit_config == "freqfit":
            self.filter_layer = GlobalFilter2D(len(self.blocks) + 1, 768, 16, 9)
        elif self.freqfit_config == "ssf":
            self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(len(self.blocks) + 1, 768)

        self.head = torch.nn.Identity()

    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]
        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)
        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            if self.freqfit_config == "freqfit":
                x = self._filter_ops(i, x)
            elif self.freqfit_config == "ssf":
                x = ssf_ada(x, self.ssf_scale[i], self.ssf_shift[i])
            x = blk(x)

        if self.freqfit_config == "freqfit":
            x = self._filter_ops(-1, x)
        elif self.freqfit_config == "ssf":
            x = ssf_ada(x, self.ssf_scale[-1], self.ssf_shift[-1])

        x_norm = self.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

class FreqFit_BOFTWrapper_DINO(BaseWrapper):
    def __init__(self, model=VisionTransformer, freqfit_config='freqfit', lora_targets=None, *args, **kwargs) -> None:
        super().__init__( model, *args, **kwargs)

        self.lora_targets = lora_targets

        self.model = VisionTransformer(freqfit_config=freqfit_config)
        self.from_pretrained()
        self.model = get_peft_model(self.model,
                                    BOFTConfig(target_modules=lora_targets))

        self.feat_dim = 768

        for name, param in self.model.named_parameters():
            if "boft" in name or 'filter' in name or 'scale' in name or 'shift' in name:# or "pos_embed" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.base_model.model.head = torch.nn.Linear(768, 2)

    def forward(self, x):
        return self.model(x)    # MedMAE


    def from_pretrained(self, path="/project/hnguyen2/stly/code/fairness/FairMedFM/pretrained/dinov2_vitb14_pretrain.pth"):
        checkpoint = torch.load(path, map_location="cpu")
        for k in list(checkpoint.keys()):
            if k.startswith('blocks.'):
                suffix = k[len("blocks."):]
                new_k = f"blocks.0.{suffix}"
                checkpoint[new_k] = checkpoint[k]
                del checkpoint[k]

        loading = self.model.load_state_dict(checkpoint, strict=False)
        print("load pretrained weight")
        print(loading)