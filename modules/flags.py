# Constants for Enabled/Disabled options
DISABLED = 'Disabled'
ENABLED = 'Enabled'
SUBTLE_VARIATION = 'Vary (Subtle)'
STRONG_VARIATION = 'Vary (Strong)'
UPSCALE_15 = 'Upscale (1.5x)'
UPSCALE_2 = 'Upscale (2x)'
UPSCALE_FAST = 'Upscale (Fast 2x)'

# Lists of available options
UOV_LIST = [DISABLED, SUBTLE_VARIATION, STRONG_VARIATION, UPSCALE_15, UPSCALE_2, UPSCALE_FAST]

SAMPLER_NAMES = {
    "euler": "Euler",
    "euler_ancestral": "Euler (Ancestral)",
    # ... other sampler names ...
    "ddpm": "DDPM"
}

SCHEDULER_NAMES = {
    "normal": "Normal",
    "karras": "Karras",
    # ... other scheduler names ...
    "ddim_uniform": "DDIM Uniform"
}

sampler_list = list(SAMPLER_NAMES.keys())
scheduler_list = list(SCHEDULER_NAMES.keys())

# Constants for Image Prompt options
CN_IP = "Image Prompt"
CN_CANNY = "PyraCanny"
CN_CPDS = "CPDS"

IP_LIST = [CN_IP, CN_CANNY, CN_CPDS]
DEFAULT_IP = CN_IP

# Default parameters (stop, weight) for Image Prompt options
DEFAULT_PARAMETERS = {
    CN_IP: (0.5, 0.6),
    CN_CANNY: (0.5, 1.0),
    CN_CPDS: (0.5, 1.0)
} #stop, weight
