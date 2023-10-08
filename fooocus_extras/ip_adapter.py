import torch
import comfy.clip_vision
import safetensors.torch as sf
import comfy.model_management as model_management
import contextlib

from fooocus_extras.resampler import Resampler
from comfy.model_patcher import ModelPatcher


if model_management.xformers_enabled():
    import xformers
    import xformers.ops


SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20


def sdp(q, k, v, extra_options):
    if model_management.xformers_enabled():
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], extra_options["n_heads"], extra_options["dim_head"])
            .permute(0, 2, 1, 3)
            .reshape(b * extra_options["n_heads"], t.shape[1], extra_options["dim_head"])
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        out = (
            out.unsqueeze(0)
            .reshape(b, extra_options["n_heads"], out.shape[1], extra_options["dim_head"])
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], extra_options["n_heads"] * extra_options["dim_head"])
        )
    else:
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
            (q, k, v),
        )
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])
    return out


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList(
            [torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])

    def load_state_dict_ordered(self, sd):
        state_dict = []
        for i in range(4096):
            for k in ['k', 'v']:
                key = f'{i}.to_{k}_ip.weight'
                if key in sd:
                    state_dict.append(sd[key])
        for i, v in enumerate(state_dict):
            self.to_kvs[i].weight = torch.nn.Parameter(v, requires_grad=False)


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, plus, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4,
                 sdxl_plus=False):
        super().__init__()
        self.plus = plus
        if self.plus:
            self.image_proj_model = Resampler(
                dim=1280 if sdxl_plus else cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if sdxl_plus else 12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4
            )
        else:
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict_ordered(state_dict["ip_adapter"])


clip_vision: comfy.clip_vision.ClipVisionModel = None
ip_negative: torch.Tensor = None
image_proj_model: ModelPatcher = None
ip_layers: ModelPatcher = None
ip_adapter: IPAdapterModel = None
ip_unconds = None


def load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path):
    global clip_vision, image_proj_model, ip_layers, ip_negative, ip_adapter, ip_unconds

    if clip_vision_path is None:
        return
    if ip_negative_path is None:
        return
    if ip_adapter_path is None:
        return
    if clip_vision is not None and image_proj_model is not None and ip_layers is not None and ip_negative is not None:
        return

    ip_negative = sf.load_file(ip_negative_path)['data']
    clip_vision = comfy.clip_vision.load(clip_vision_path)

    load_device = model_management.get_torch_device()
    offload_device = torch.device('cpu')

    use_fp16 = model_management.should_use_fp16(device=load_device)
    ip_state_dict = torch.load(ip_adapter_path, map_location="cpu")
    plus = "latents" in ip_state_dict["image_proj"]
    cross_attention_dim = ip_state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    sdxl = cross_attention_dim == 2048
    sdxl_plus = sdxl and plus

    if plus:
        clip_extra_context_tokens = ip_state_dict["image_proj"]["latents"].shape[1]
        clip_embeddings_dim = ip_state_dict["image_proj"]["latents"].shape[2]
    else:
        clip_extra_context_tokens = ip_state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim
        clip_embeddings_dim = None

    ip_adapter = IPAdapterModel(
        ip_state_dict,
        plus=plus,
        cross_attention_dim=cross_attention_dim,
        clip_embeddings_dim=clip_embeddings_dim,
        clip_extra_context_tokens=clip_extra_context_tokens,
        sdxl_plus=sdxl_plus
    )
    ip_adapter.sdxl = sdxl
    ip_adapter.load_device = load_device
    ip_adapter.offload_device = offload_device
    ip_adapter.dtype = torch.float16 if use_fp16 else torch.float32
    ip_adapter.to(offload_device, dtype=ip_adapter.dtype)

    image_proj_model = ModelPatcher(model=ip_adapter.image_proj_model, load_device=load_device,
                                    offload_device=offload_device)
    ip_layers = ModelPatcher(model=ip_adapter.ip_layers, load_device=load_device,
                             offload_device=offload_device)

    ip_unconds = None
    return


@torch.no_grad()
@torch.inference_mode()
def preprocess(img):
    global ip_unconds

    inputs = clip_vision.processor(images=img, return_tensors="pt")
    comfy.model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = inputs['pixel_values'].to(clip_vision.load_device)

    if clip_vision.dtype != torch.float32:
        precision_scope = torch.autocast
    else:
        precision_scope = lambda a, b: contextlib.nullcontext(a)

    with precision_scope(comfy.model_management.get_autocast_device(clip_vision.load_device), torch.float32):
        outputs = clip_vision.model(pixel_values=pixel_values, output_hidden_states=True)

    if ip_adapter.plus:
        cond = outputs.hidden_states[-2].to(ip_adapter.dtype)
    else:
        cond = outputs.image_embeds.to(ip_adapter.dtype)

    comfy.model_management.load_model_gpu(image_proj_model)
    cond = image_proj_model.model(cond).to(device=ip_adapter.load_device, dtype=ip_adapter.dtype)

    comfy.model_management.load_model_gpu(ip_layers)

    if ip_unconds is None:
        uncond = ip_negative.to(device=ip_adapter.load_device, dtype=ip_adapter.dtype)
        ip_unconds = [m(uncond).cpu() for m in ip_layers.model.to_kvs]

    ip_conds = [m(cond).cpu() for m in ip_layers.model.to_kvs]
    return ip_conds


@torch.no_grad()
@torch.inference_mode()
def patch_model(model, tasks):
    new_model = model.clone()

    def make_attn_patcher(ip_index):
        def patcher(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            current_step = float(model.model.diffusion_model.current_step.detach().cpu().numpy()[0])
            cond_or_uncond = extra_options['cond_or_uncond']

            with torch.autocast("cuda", dtype=ip_adapter.dtype):
                q = n
                k = [context_attn2]
                v = [value_attn2]
                b, _, _ = q.shape

                for ip_conds, cn_stop, cn_weight in tasks:
                    if current_step < cn_stop:
                        ip_k_c = ip_conds[ip_index * 2].to(q)
                        ip_v_c = ip_conds[ip_index * 2 + 1].to(q)
                        ip_k_uc = ip_unconds[ip_index * 2].to(q)
                        ip_v_uc = ip_unconds[ip_index * 2 + 1].to(q)

                        ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0)
                        ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0)

                        # Midjourney's attention formulation of image prompt (non-official reimplementation)
                        # Written by Lvmin Zhang at Stanford University, 2023 Dec
                        # For non-commercial use only - if you use this in commercial project then
                        # probably it has some intellectual property issues.
                        # Contact lvminzhang@acm.org if you are not sure.

                        # Below is the sensitive part with potential intellectual property issues.

                        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                        ip_v_offset = ip_v - ip_v_mean

                        B, F, C = ip_k.shape
                        channel_penalty = float(C) / 1280.0
                        weight = cn_weight * channel_penalty

                        ip_k = ip_k * weight
                        ip_v = ip_v_offset + ip_v_mean * weight

                        k.append(ip_k)
                        v.append(ip_v)

                k = torch.cat(k, dim=1)
                v = torch.cat(v, dim=1)
                out = sdp(q, k, v, extra_options)

            return out.to(dtype=org_dtype)
        return patcher

    def set_model_patch_replace(model, number, key):
        to = model.model_options["transformer_options"]
        if "patches_replace" not in to:
            to["patches_replace"] = {}
        if "attn2" not in to["patches_replace"]:
            to["patches_replace"]["attn2"] = {}
        if key not in to["patches_replace"]["attn2"]:
            to["patches_replace"]["attn2"][key] = make_attn_patcher(number)

    number = 0
    if not ip_adapter.sdxl:
        for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
            set_model_patch_replace(new_model, number, ("input", id))
            number += 1
        for id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # id of output_blocks that have cross attention
            set_model_patch_replace(new_model, number, ("output", id))
            number += 1
        set_model_patch_replace(new_model, number, ("middle", 0))
    else:
        for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                set_model_patch_replace(new_model, number, ("input", id, index))
                number += 1
        for id in range(6):  # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                set_model_patch_replace(new_model, number, ("output", id, index))
                number += 1
        for index in range(10):
            set_model_patch_replace(new_model, number, ("middle", 0, index))
            number += 1

    return new_model
