import torch
import fcbh.samplers
import fcbh.model_management

from fcbh.model_base import SDXLRefiner, SDXL
from fcbh.sample import get_additional_models, get_models_from_cond, cleanup_additional_models
from fcbh.samplers import resolve_areas_and_cond_masks, wrap_model, calculate_start_end_timesteps, \
    create_cond_with_same_area_if_none, pre_run_control, apply_empty_x_to_equal_area, encode_adm, \
    encode_cond


current_refiner = None
refiner_switch_step = -1


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    c, p = cond[0]
    if target_model is None or isinstance(target_model, SDXLRefiner):
        c = c[..., -1280:].clone()
        p = {"pooled_output": p["pooled_output"].clone()}
    elif isinstance(target_model, SDXL):
        c = c.clone()
        p = {"pooled_output": p["pooled_output"].clone()}
    else:
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)

        p = {}
    return [[c, p]]


@torch.no_grad()
@torch.inference_mode()
def sample_hacked(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    global current_refiner

    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model_wrap, negative)
    calculate_start_end_timesteps(model_wrap, positive)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    # pre_run_control(model_wrap, negative + positive)
    pre_run_control(model_wrap, positive)  # negative is not necessary in Fooocus, 0.5s faster.

    apply_empty_x_to_equal_area(list(filter(lambda c: c[1].get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if model.is_adm():
        positive = encode_adm(model, positive, noise.shape[0], noise.shape[3], noise.shape[2], device, "positive")
        negative = encode_adm(model, negative, noise.shape[0], noise.shape[3], noise.shape[2], device, "negative")

    if hasattr(model, 'cond_concat'):
        positive = encode_cond(model.cond_concat, "concat", positive, device, noise=noise, latent_image=latent_image, denoise_mask=denoise_mask)
        negative = encode_cond(model.cond_concat, "concat", negative, device, noise=noise, latent_image=latent_image, denoise_mask=denoise_mask)

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    if current_refiner is not None and current_refiner.model.is_adm():
        positive_refiner = clip_separate(positive, target_model=current_refiner.model)
        negative_refiner = clip_separate(negative, target_model=current_refiner.model)

        positive_refiner = encode_adm(current_refiner.model, positive_refiner, noise.shape[0], noise.shape[3], noise.shape[2], device, "positive")
        negative_refiner = encode_adm(current_refiner.model, negative_refiner, noise.shape[0], noise.shape[3], noise.shape[2], device, "negative")

        positive_refiner[0][1]['adm_encoded'].to(positive[0][1]['adm_encoded'])
        negative_refiner[0][1]['adm_encoded'].to(negative[0][1]['adm_encoded'])

    def refiner_switch():
        cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))

        extra_args["cond"] = positive_refiner
        extra_args["uncond"] = negative_refiner

        # clear ip-adapter for refiner
        extra_args['model_options'] = {k: {} if k == 'transformer_options' else v for k, v in extra_args['model_options'].items()}

        models, inference_memory = get_additional_models(positive_refiner, negative_refiner, current_refiner.model_dtype())
        fcbh.model_management.load_models_gpu([current_refiner] + models, fcbh.model_management.batch_area_memory(
            noise.shape[0] * noise.shape[2] * noise.shape[3]) + inference_memory)

        model_wrap.inner_model.inner_model = current_refiner.model
        print('Refiner Swapped')
        return

    def callback_wrap(step, x0, x, total_steps):
        if step == refiner_switch_step and current_refiner is not None:
            refiner_switch()
        if callback is not None:
            # residual_noise_preview = x - x0
            # residual_noise_preview /= residual_noise_preview.std()
            # residual_noise_preview *= x0.std()
            callback(step, x0, x, total_steps)

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback_wrap, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))


fcbh.samplers.sample = sample_hacked
