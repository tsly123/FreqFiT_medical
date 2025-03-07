import json

import torch

import models
from utils.static import CLIP_MODELS
from utils.tokenizer import tokenize_text



def get_warpped_model(args, model, data_engine=None):

    if args.task == "cls":
        if args.usage == "lp":
            from wrappers import LinearProbeWrapper

            model_warpped = LinearProbeWrapper(model)
        elif args.usage in ["clip-zs", "clip-adapt"]:
            assert args.model in CLIP_MODELS, f"{args.usage} is not applicable for {args.model}"
            from wrappers import CLIPWrapper

            text = tokenize_text(args, args.data_setting["class_names"])
            text_features = model.encode_text(text.to(args.device))

            model_warpped = CLIPWrapper(model, text_features)

        elif args.usage in ["lora", 'boft', 'adalora', 'fourierft'] and args.tune_method not in ['freqfit', 'ssf']:
            if args.model == "MedMAE":
                if args.usage == 'lora':
                    from wrappers import LoRAWrapper as PEFTWrapper
                elif args.usage == 'boft':
                    from wrappers import BOFTWrapper as PEFTWrapper
                elif args.usage == 'adalora':
                    from wrappers import AdaLoRAWrapper as PEFTWrapper
                elif args.usage == 'fourierft':
                    from wrappers import FourierFTWrapper as PEFTWrapper
                model_setting = args.model_setting

                assert (
                    "lora_targets" in model_setting.keys()
                ), f"LoRA is not applicable for {args.model}, either because it's not a ViT-based model or it's not supported in the current version"

                model_warpped = PEFTWrapper(
                    model,
                    lora_targets=model_setting["lora_targets"])

            elif args.model == "DINOv2":
                if args.usage == 'lora':
                    from wrappers import LoRAWrapper_DINO as PEFTWrapper
                elif args.usage == 'boft':
                    from wrappers import BOFTWrapper_DINO as PEFTWrapper
                elif args.usage == 'adalora':
                    from wrappers import AdaLoRAWrapper_DINO as PEFTWrapper
                elif args.usage == 'fourierft':
                    from wrappers import FourierFTWrapper_DINO as PEFTWrapper
                model_setting = args.model_setting

                assert (
                    "lora_targets" in model_setting.keys()
                ), f"LoRA is not applicable for {args.model}, either because it's not a ViT-based model or it's not supported in the current version"

                model_warpped = PEFTWrapper(
                    model,
                    lora_targets=model_setting["lora_targets"])

        elif args.usage in ["lora", 'boft', 'adalora', 'fourierft'] and args.tune_method in ['freqfit', 'ssf']:
            if args.model == "MedMAE":
                if args.usage == 'lora':
                    from wrappers import FreqFit_LoRAWrapper as FreqFitWrapper
                elif args.usage == 'boft':
                    from wrappers import FreqFit_BOFTWrapper as FreqFitWrapper
                elif args.usage == 'adalora':
                    from wrappers import FreqFit_AdaLoRAWrapper as FreqFitWrapper
                elif args.usage == 'fourierft':
                    from wrappers import FreqFit_FourierFTWrapper as FreqFitWrapper


                model_setting = args.model_setting

                assert (
                    "lora_targets" in model_setting.keys()
                ), f"LoRA is not applicable for {args.model}, either because it's not a ViT-based model or it's not supported in the current version"

                model_warpped = FreqFitWrapper(model,
                                               freqfit_config=args.tune_method,
                                               lora_targets=model_setting["lora_targets"])

            elif args.model == "DINOv2":
                if args.usage == 'lora':
                    from wrappers import FreqFit_LoRAWrapper_DINO as FreqFitWrapper
                elif args.usage == 'boft':
                    from wrappers import FreqFit_BOFTWrapper_DINO as FreqFitWrapper
                elif args.usage == 'adalora':
                    from wrappers import FreqFit_AdaLoRAWrapper_DINO as FreqFitWrapper
                elif args.usage == 'fourierft':
                    from wrappers import FreqFit_FourierFTWrapper_DINO as FreqFitWrapper

                model_setting = args.model_setting

                assert (
                        "lora_targets" in model_setting.keys()
                ), f"LoRA is not applicable for {args.model}, either because it's not a ViT-based model or it's not supported in the current version"

                model_warpped = FreqFitWrapper(model,
                                               freqfit_config=args.tune_method,
                                               lora_targets=model_setting["lora_targets"])
        else:
            raise NotImplementedError()

    elif args.task == "seg":
        from wrappers import SAMWrapper
        model_warpped = SAMWrapper(model, data_engine=data_engine)
    else:
        raise NotImplementedError

    return model_warpped
