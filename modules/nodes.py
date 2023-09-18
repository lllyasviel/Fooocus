import copy
import torch
from functools import wraps

# import modules.default_pipeline as pipeline

from .sdxl_styles import styles, apply_style, aspect_ratios, fooocus_expansion
from .util import join_prompts, remove_empty_str
from .core import ksampler_with_refiner
from .model_loader import load_file_from_url
from .path import fooocus_expansion_path
from .expansion import (
    safe_str,
    fooocus_magic_split,
    dangrous_patterns,
    remove_pattern,
    FooocusExpansion,
)
from modules.patch import (
    patched_discrete_eps_ddpm_denoiser_forward,
    sdxl_encode_adm_patched,
    cfg_patched,
)
import modules.patch

# within comfy context
import comfy
import folder_paths


NODE_CLASS_MAPPINGS = {}


def add_to_node(N):
    NODE_CLASS_MAPPINGS[N.__name__] = N
    return N


available_styles = [fooocus_expansion] + [i for i in styles]


# expansion = FooocusExpansion()


@add_to_node
class FooocusExpansionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("EXPAND_MODEL",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "fooocus/loaders"

    def load_checkpoint(self):
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
            model_dir=fooocus_expansion_path,
            file_name="pytorch_model.bin",
        )
        expansion = FooocusExpansion()
        return (expansion,)


@add_to_node
class FooocusEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anytype": ("OUT",)}}

    RETURN_TYPES = ()
    FUNCTION = "entry"
    OUTPUT_NODE = True

    CATEGORY = "fooocus"

    def entry(self, anytype):
        return {
            "ui": {
                "images": [
                    {
                        "filename": "ComfyUI_00004_.png",
                        "subfolder": "",
                        "type": "output",
                    }
                ]
            }
        }


@add_to_node
class FooocusStyler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"current_select": (available_styles,)},
            "optional": {"selected": ("LIST",)},
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "append_style"

    CATEGORY = "fooocus/styler"

    def append_style(self, current_select, selected=[]):
        selected = [i for i in selected]
        selected.append(current_select)
        return (selected,)


@add_to_node
class FooocusStyleMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EXPAND_MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1 << 32 - 1}),
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
            },
            "optional": {"selected": ("LIST",)},
        }

    RETURN_TYPES = ("LIST", "LIST", "INT", "INT")
    FUNCTION = "entry"
    OUTPUT_NODE = True

    CATEGORY = "fooocus/styler"

    def entry(
        self,
        model,
        seed,
        prompt,
        negative_prompt,
        selected=[],
    ):
        style_selections = [i for i in selected]
        raw_style_selections = [i for i in style_selections]

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        # modules.patch.sharpness = sharpness

        raw_prompt = prompt
        raw_negative_prompt = negative_prompt

        prompts = remove_empty_str(
            [safe_str(p) for p in prompt.split("\n")], default=""
        )
        negative_prompts = remove_empty_str(
            [safe_str(p) for p in negative_prompt.split("\n")], default=""
        )

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = (
            negative_prompts[1:] if len(negative_prompts) > 1 else []
        )

        # pipeline.refresh_base_model(base_model_name)
        # pipeline.refresh_refiner_model(refiner_model_name)
        # pipeline.refresh_loras(loras)
        # pipeline.clear_all_caches()

        positive_basic_workloads = []
        negative_basic_workloads = []

        if use_style:
            for s in style_selections:
                p, n = apply_style(s, positive=prompt)
                positive_basic_workloads.append(p)
                negative_basic_workloads.append(n)
        else:
            positive_basic_workloads.append(prompt)

        negative_basic_workloads.append(
            negative_prompt
        )  # Always use independent workload for negative.

        positive_basic_workloads = positive_basic_workloads + extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + extra_negative_prompts

        positive_basic_workloads = remove_empty_str(
            positive_basic_workloads, default=prompt
        )
        negative_basic_workloads = remove_empty_str(
            negative_basic_workloads, default=negative_prompt
        )

        positive_top_k = len(positive_basic_workloads)
        negative_top_k = len(negative_basic_workloads)

        tasks = [
            dict(
                task_seed=seed,
                positive=positive_basic_workloads,
                negative=negative_basic_workloads,
                expansion="",
                c=[None, None],
                uc=[None, None],
            )
        ]
        if use_expansion:
            for t in tasks:
                expansion = model(prompt, t["task_seed"])
                t["expansion"] = expansion
                t["positive"] = copy.deepcopy(t["positive"]) + [
                    join_prompts(prompt, expansion)
                ]  # Deep copy.
        else:
            t = tasks[0]
        return t["positive"], t["negative"], positive_top_k, negative_top_k


@add_to_node
class FooocusCLIPEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texts": ("LIST",),
                "clip": ("CLIP",),
                "pool_top_k": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "fooocus/conditioning"

    def encode(self, texts, clip, pool_top_k=1):
        cond_list = []
        pooled_acc = 0
        for i, text in enumerate(texts):
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            if i < pool_top_k:
                pooled_acc += pooled
        return ([[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]],)


@add_to_node
class FooocusKsamplerWithRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "refiner": ("MODEL",),
                "refiner_positive": ("CONDITIONING",),
                "refiner_negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "refiner_switch_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "dpmpp_2m_sde_gpu"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "karras"},
                ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "disable_noise": (["enable", "disable"], {"default": "disable"}),
                "start_step": ("INT", {"default": 0, "min": -1, "max": 10000}),
                "last_step": ("INT", {"default": 10000, "min": -1, "max": 10000}),
                "force_full_denoise": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "fooocus/sampling"

    def sample(
        self,
        model,
        positive,
        negative,
        refiner,
        refiner_positive,
        refiner_negative,
        latent,
        seed,
        steps,
        refiner_switch_step,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        disable_noise,
        start_step,
        last_step,
        force_full_denoise,
        callback_function=None,
    ):
        disable_noise = disable_noise == "enable"
        force_full_denoise = force_full_denoise == "enable"
        out = ksampler_with_refiner(
            model,
            positive,
            negative,
            refiner,
            refiner_positive,
            refiner_negative,
            latent,
            seed,
            steps,
            refiner_switch_step,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            disable_noise,
            start_step,
            last_step,
            force_full_denoise,
            callback_function=callback_function,
        )
        return ({"samples": out["samples"]},)


@add_to_node
class FooocusGlobalStateDoPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sharpness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1 << 32 - 1}),
            },
            "optional": {"refiner": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    FUNCTION = "set_fooocus_model"

    CATEGORY = "fooocus/utils"

    def set_fooocus_model(self, model, sharpness, seed, refiner=None):
        modules.patch.sharpness = sharpness
        model.model_options["sampler_cfg_function"] = cfg_patched
        if refiner is not None:
            refiner.model_options["sampler_cfg_function"] = cfg_patched
        if not hasattr(
            comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser, "old_forward"
        ):
            comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.old_forward = (
                comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.forward
            )
            comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.forward = (
                patched_discrete_eps_ddpm_denoiser_forward
            )
        if not hasattr(model.model, "old_encode_adm"):
            model.model.old_encode_adm = model.model.encode_adm
            model.model.encode_adm = sdxl_encode_adm_patched.__get__(model.model)
        return (model, refiner)


@add_to_node
class FooocusGlobalStateUndoPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1 << 32 - 1}),
            },
            "optional": {"refiner": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL", "LATENT", "MODEL")
    FUNCTION = "unset_fooocus_model"

    CATEGORY = "fooocus/utils"

    def unset_fooocus_model(self, model, latent, seed, refiner=None):
        del model.model_options["sampler_cfg_function"]
        if refiner is not None:
            del refiner.model_options["sampler_cfg_function"]
        if hasattr(comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser, "old_forward"):
            comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.forward = (
                comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.old_forward
            )
            delattr(comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser, "old_forward")
        if hasattr(model.model, "old_encode_adm"):
            model.model.encode_adm = model.model.old_encode_adm
            delattr(model.model, "old_encode_adm")
        return (model, latent, refiner)


@add_to_node
class FooocusLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "current_select": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {"selected": ("LIST",)},
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "append_lora"

    CATEGORY = "fooocus/loaders"

    def append_lora(self, current_select, strength, selected=[]):
        selected = [i for i in selected]
        selected.append((current_select, strength))
        return (selected,)


@add_to_node
class FooocusLoraLoader:
    def __init__(self):
        self.loaded_lora = dict()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "lora_names": ("LIST",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "fooocus/loaders"

    def load_lora(self, model, clip, lora_names=[]):
        model_lora, clip_lora = model, clip
        for lora_name, strength in lora_names:
            if lora_name is None:
                continue
            strength_model = strength_clip = strength
            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = None
            if lora_path in self.loaded_lora:
                lora = self.loaded_lora[lora_path]

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora[lora_path] = lora
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model_lora, clip_lora, lora, strength_model, strength_clip
            )
        return (model_lora, clip_lora)
