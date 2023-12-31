import torch
import ldm_patched.modules.clip_vision
import safetensors.torch as sf
import ldm_patched.modules.model_management as model_management
import ldm_patched.ldm.modules.attention as attention

from extras.resampler import Resampler
from ldm_patched.modules.model_patcher import ModelPatcher
from modules.core import numpy_to_pytorch
from modules.ops import use_patched_ops
from ldm_patched.modules.ops import manual_cast


SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20


def sdp(q, k, v, extra_options):
    return attention.optimized_attention(q, k, v, heads=extra_options["n_heads"], mask=None)


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


clip_vision: ldm_patched.modules.clip_vision.ClipVisionModel = None
ip_negative: torch.Tensor = None
ip_adapters: dict = {}


def load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path):
    global clip_vision, ip_negative, ip_adapters

    if clip_vision is None and isinstance(clip_vision_path, str):
        clip_vision = ldm_patched.modules.clip_vision.load(clip_vision_path)

    if ip_negative is None and isinstance(ip_negative_path, str):
        ip_negative = sf.load_file(ip_negative_path)['data']

    if not isinstance(ip_adapter_path, str) or ip_adapter_path in ip_adapters:
        return

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

    with use_patched_ops(manual_cast):
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

    ip_adapters[ip_adapter_path] = dict(
        ip_adapter=ip_adapter,
        image_proj_model=image_proj_model,
        ip_layers=ip_layers,
        ip_unconds=None
    )

    return


@torch.no_grad()
@torch.inference_mode()
def clip_preprocess(image):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype).view([1, 3, 1, 1])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype).view([1, 3, 1, 1])
    image = image.movedim(-1, 1)

    # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
    B, C, H, W = image.shape
    assert H == 224 and W == 224

    return (image - mean) / std


@torch.no_grad()
@torch.inference_mode()
def preprocess(img, ip_adapter_path):
    global ip_adapters
    entry = ip_adapters[ip_adapter_path]

    ldm_patched.modules.model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_preprocess(numpy_to_pytorch(img).to(clip_vision.load_device))
    outputs = clip_vision.model(pixel_values=pixel_values, output_hidden_states=True)

    ip_adapter = entry['ip_adapter']
    ip_layers = entry['ip_layers']
    image_proj_model = entry['image_proj_model']
    ip_unconds = entry['ip_unconds']

    if ip_adapter.plus:
        cond = outputs.hidden_states[-2]
    else:
        cond = outputs.image_embeds

    cond = cond.to(device=ip_adapter.load_device, dtype=ip_adapter.dtype)

    ldm_patched.modules.model_management.load_model_gpu(image_proj_model)
    cond = image_proj_model.model(cond).to(device=ip_adapter.load_device, dtype=ip_adapter.dtype)

    ldm_patched.modules.model_management.load_model_gpu(ip_layers)

    if ip_unconds is None:
        uncond = ip_negative.to(device=ip_adapter.load_device, dtype=ip_adapter.dtype)
        ip_unconds = [m(uncond).cpu() for m in ip_layers.model.to_kvs]
        entry['ip_unconds'] = ip_unconds

    ip_conds = [m(cond).cpu() for m in ip_layers.model.to_kvs]

    return ip_conds, ip_unconds


@torch.no_grad()
@torch.inference_mode()
def patch_model(model, tasks):
    new_model = model.clone()

    def make_attn_patcher(ip_index):
        def patcher(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            current_step = float(model.model.diffusion_model.current_step.detach().cpu().numpy()[0])
            cond_or_uncond = extra_options['cond_or_uncond']

            q = n
            k = [context_attn2]
            v = [value_attn2]
            b, _, _ = q.shape

            for (cs, ucs), cn_stop, cn_weight in tasks:
                if current_step < cn_stop:
                    ip_k_c = cs[ip_index * 2].to(q)
                    ip_v_c = cs[ip_index * 2 + 1].to(q)
                    ip_k_uc = ucs[ip_index * 2].to(q)
                    ip_v_uc = ucs[ip_index * 2 + 1].to(q)

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

    for id in [4, 5, 7, 8]:
        block_indices = range(2) if id in [4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ("input", id, index))
            number += 1

    for id in range(6):
        block_indices = range(2) if id in [3, 4, 5] else range(10)
        for index in block_indices:
            set_model_patch_replace(new_model, number, ("output", id, index))
            number += 1

    for index in range(10):
        set_model_patch_replace(new_model, number, ("middle", 0, index))
        number += 1

    return new_model
