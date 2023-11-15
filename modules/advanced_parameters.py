disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,  \
    scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
    overwrite_vary_strength, overwrite_upscale_strength, \
    mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
    debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, inpaint_engine, \
    refiner_swap_method, \
    freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = [None] * 28


def set_all_advanced_parameters(*args):
    global disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
        overwrite_vary_strength, overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, inpaint_engine, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
        scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height, \
        overwrite_vary_strength, overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, inpaint_engine, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = args

    return
