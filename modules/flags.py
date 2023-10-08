import comfy.samplers


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

sampler_list = comfy.samplers.SAMPLER_NAMES
default_sampler = 'dpmpp_2m_sde_gpu'

scheduler_list = comfy.samplers.SCHEDULER_NAMES
default_scheduler = "karras"

cn_ip = "Image Prompt"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_canny, cn_cpds]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.4, 0.6), cn_canny: (0.4, 1.0), cn_cpds: (0.4, 1.0)
}  # stop, weight
