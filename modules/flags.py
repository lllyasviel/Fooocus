disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'

uov_list = [
    disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast
]

sampler_list = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
                # "ddim",
                "uni_pc", "uni_pc_bh2",
                # "dpmpp_fooocus_2m_sde_inpaint_seamless"
                ]
default_sampler = 'dpmpp_2m_sde_gpu'

scheduler_list = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
default_scheduler = "karras"

ip_list = ["Image Prompt", "Structure"]
default_ip = "Image Prompt"
ip_number = 4
