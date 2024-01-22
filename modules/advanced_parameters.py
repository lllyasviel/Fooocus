overwrite_upscale_strength, \
    mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
    debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
    refiner_swap_method, \
    freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
    debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
    inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate = [None] * 23


def set_all_advanced_parameters(*args):
    global overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
        inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate

    overwrite_upscale_strength, \
        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
        debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold, \
        refiner_swap_method, \
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, \
        inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate = args

    return
