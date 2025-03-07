import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from wrappers.base import BaseWrapper
import math
import timm
from timm.models.vision_transformer import Block, Attention
from peft import AdaLoraConfig, get_peft_model

from wrappers.freqfit import GlobalFilter2D, init_ssf_scale_shift, ssf_ada

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, freqfit_config=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.freqfit_config = freqfit_config

        if self.freqfit_config == "freqfit":
            self.filter_layer = GlobalFilter2D(len(self.blocks)+1, 768)
        elif self.freqfit_config == "ssf":
            self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(len(self.blocks)+1, 768)


        self.head = torch.nn.Identity()

    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]
        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)
        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

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

        x = self.norm(x)

        return x


class FreqFit_AdaLoRAWrapper(BaseWrapper):
    def __init__(self, model=VisionTransformer, freqfit_config='freqfit', lora_targets=None, *args, **kwargs) -> None:
        super().__init__( model, *args, **kwargs)

        self.lora_targets = lora_targets
        self.model = get_peft_model(VisionTransformer(freqfit_config=freqfit_config),
                                    AdaLoraConfig(target_modules=lora_targets))
        self.from_pretrained(path="./pretrained/medmae/vit-b_CXR_0.5M_mae.pth")

        self.feat_dim = 768

        for name, param in self.model.named_parameters():
            if "lora" in name or 'filter' in name or 'scale' in name or 'shift' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.base_model.model.head = torch.nn.Linear(768, 2)

    def forward(self, x):
        return self.model(x)    # MedMAE


    def from_pretrained(self, path="./pretrained/medmae/vit-b_CXR_0.5M_mae.pth"):
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = self.model.model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        for k in list(checkpoint_model.keys()):
            if 'decoder' in k:
                del checkpoint_model[k]
                continue
            if 'attn.qkv' in k or 'attn.proj' in k:
                if '.weight' in k:
                    new_k = f"{k[:-len('.weight')]}.base_layer.weight"
                elif '.bias' in k:
                    new_k = f"{k[:-len('.bias')]}.base_layer.bias"
                checkpoint_model[new_k] = checkpoint_model[k]
                del checkpoint_model[k]

        loading = self.model.model.load_state_dict(checkpoint_model, strict=False)
        print("load pretrained weight MAE FreqFit")
        print(loading)