import torch
import comfy.model_base
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.samplers
import comfy.k_diffusion.external
import comfy.model_management
import modules.anisotropic as anisotropic
import comfy.ldm.modules.attention
import comfy.sd1_clip

from comfy.k_diffusion import utils


sharpness = 2.0

cfg_x0 = 0.0
cfg_s = 1.0


def cfg_patched(args):
    global cfg_x0, cfg_s
    positive_eps = args['cond'].clone()
    positive_x0 = args['cond'] * cfg_s + cfg_x0
    uncond = args['uncond'] * cfg_s + cfg_x0
    cond_scale = args['cond_scale']
    t = args['timestep']

    alpha = 1.0 - (t / 999.0)[:, None, None, None].clone()
    alpha *= 0.001 * sharpness

    eps_degraded = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=positive_x0)
    eps_degraded_weighted = eps_degraded * alpha + positive_eps * (1.0 - alpha)

    cond = eps_degraded_weighted * cfg_s + cfg_x0

    return uncond + (cond - uncond) * cond_scale


def patched_discrete_eps_ddpm_denoiser_forward(self, input, sigma, **kwargs):
    global cfg_x0, cfg_s
    c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
    cfg_x0 = input
    cfg_s = c_out
    return self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = kwargs["pooled_output"]
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    crop_w = kwargs.get("crop_w", 0)
    crop_h = kwargs.get("crop_h", 0)
    target_width = kwargs.get("target_width", width)
    target_height = kwargs.get("target_height", height)

    if kwargs.get("prompt_type", "") == "negative":
        width *= 0.8
        height *= 0.8
    elif kwargs.get("prompt_type", "") == "positive":
        width *= 1.5
        height *= 1.5

    out = []
    out.append(self.embedder(torch.Tensor([height])))
    out.append(self.embedder(torch.Tensor([width])))
    out.append(self.embedder(torch.Tensor([crop_h])))
    out.append(self.embedder(torch.Tensor([crop_w])))
    out.append(self.embedder(torch.Tensor([target_height])))
    out.append(self.embedder(torch.Tensor([target_width])))
    flat = torch.flatten(torch.cat(out))[None, ]
    return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def sdxl_refiner_encode_adm_patched(self, **kwargs):
    clip_pooled = kwargs["pooled_output"]
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    crop_w = kwargs.get("crop_w", 0)
    crop_h = kwargs.get("crop_h", 0)

    if kwargs.get("prompt_type", "") == "negative":
        aesthetic_score = kwargs.get("aesthetic_score", 2.5)
    else:
        aesthetic_score = kwargs.get("aesthetic_score", 7.0)

    if kwargs.get("prompt_type", "") == "negative":
        width *= 0.8
        height *= 0.8
    elif kwargs.get("prompt_type", "") == "positive":
        width *= 1.5
        height *= 1.5

    out = []
    out.append(self.embedder(torch.Tensor([height])))
    out.append(self.embedder(torch.Tensor([width])))
    out.append(self.embedder(torch.Tensor([crop_h])))
    out.append(self.embedder(torch.Tensor([crop_w])))
    out.append(self.embedder(torch.Tensor([aesthetic_score])))
    flat = torch.flatten(torch.cat(out))[None,]
    return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def text_encoder_device_patched():
    # Fooocus's style system uses text encoder much more times than comfy so this makes things much faster.
    return comfy.model_management.get_torch_device()


def encode_token_weights_patched_with_a1111_method(self, token_weight_pairs):
    to_encode = list(self.empty_tokens)
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        to_encode.append(tokens)

    out, pooled = self.encode(to_encode)

    z_empty = out[0:1]
    if pooled.shape[0] > 1:
        first_pooled = pooled[1:2]
    else:
        first_pooled = pooled[0:1]

    output = []
    for k in range(1, out.shape[0]):
        z = out[k:k + 1]
        original_mean = z.mean()

        for i in range(len(z)):
            for j in range(len(z[i])):
                weight = token_weight_pairs[k - 1][j][1]
                z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]

        new_mean = z.mean()
        z = z * (original_mean / new_mean)
        output.append(z)

    if len(output) == 0:
        return z_empty.cpu(), first_pooled.cpu()
    return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()


def patch_all():
    comfy.ldm.modules.attention.print = lambda x: None

    comfy.model_management.text_encoder_device = text_encoder_device_patched
    print(f'Fooocus Text Processing Pipelines are retargeted to {str(comfy.model_management.text_encoder_device())}')

    comfy.k_diffusion.external.DiscreteEpsDDPMDenoiser.forward = patched_discrete_eps_ddpm_denoiser_forward
    comfy.model_base.SDXL.encode_adm = sdxl_encode_adm_patched
    # comfy.model_base.SDXLRefiner.encode_adm = sdxl_refiner_encode_adm_patched

    comfy.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = encode_token_weights_patched_with_a1111_method
    return
