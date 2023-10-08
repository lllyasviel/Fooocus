import torch
import comfy.samplers
import comfy.model_management

from comfy.sample import prepare_sampling, cleanup_additional_models, get_additional_models
from comfy.samplers import resolve_areas_and_cond_masks, wrap_model, calculate_start_end_timesteps, \
    create_cond_with_same_area_if_none, pre_run_control, apply_empty_x_to_equal_area, encode_adm, \
    blank_inpaint_image_like


current_refiner = None
refiner_switch_step = -1


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond):
    c, p = cond[0]
    c = c[..., -1280:].clone()
    p = p["pooled_output"].clone()
    return [[c, {"pooled_output": p}]]


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

    if model.is_adm():
        positive = encode_adm(model, positive, noise.shape[0], noise.shape[3], noise.shape[2], device, "positive")
        negative = encode_adm(model, negative, noise.shape[0], noise.shape[3], noise.shape[2], device, "negative")

    if current_refiner is not None and current_refiner.model.is_adm():
        positive_refiner = encode_adm(current_refiner.model, clip_separate(positive), noise.shape[0], noise.shape[3], noise.shape[2], device, "positive")
        negative_refiner = encode_adm(current_refiner.model, clip_separate(negative), noise.shape[0], noise.shape[3], noise.shape[2], device, "negative")

        positive_refiner[0][1]['adm_encoded'].to(positive[0][1]['adm_encoded'])
        negative_refiner[0][1]['adm_encoded'].to(negative[0][1]['adm_encoded'])

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    extra_args = {"cond": positive, "uncond": negative, "cond_scale": cfg, "model_options": model_options, "seed": seed}

    cond_concat = None
    if hasattr(model, 'concat_keys'):  # inpaint
        cond_concat = []
        for ck in model.concat_keys:
            if denoise_mask is not None:
                if ck == "mask":
                    cond_concat.append(denoise_mask[:,:1])
                elif ck == "masked_image":
                    cond_concat.append(latent_image) #NOTE: the latent_image should be masked by the mask in pixel space
            else:
                if ck == "mask":
                    cond_concat.append(torch.ones_like(noise)[:, :1])
                elif ck == "masked_image":
                    cond_concat.append(blank_inpaint_image_like(noise))
        extra_args["cond_concat"] = cond_concat

    def refiner_switch():
        extra_args["cond"] = positive_refiner
        extra_args["uncond"] = negative_refiner

        # clear ip-adapter for refiner
        extra_args['model_options'] = {k: {} if k == 'transformer_options' else v for k, v in extra_args['model_options'].items()}

        models, inference_memory = get_additional_models(positive_refiner, negative_refiner, current_refiner.model_dtype())
        comfy.model_management.load_models_gpu([current_refiner] + models, comfy.model_management.batch_area_memory(
            noise.shape[0] * noise.shape[2] * noise.shape[3]) + inference_memory)

        model_wrap.inner_model.inner_model = current_refiner.model
        print('Refiner Swapped')
        return

    def callback_wrap(step, x0, x, total_steps):
        if step == refiner_switch_step and current_refiner is not None:
            refiner_switch()
        if callback is not None:
            callback(step, x0, x, total_steps)

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback_wrap, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))


comfy.samplers.sample = sample_hacked
