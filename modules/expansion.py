import torch
import comfy.model_management as model_management

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from modules.path import fooocus_expansion_path
from comfy.sd import ModelPatcher


fooocus_magic_split = [
    ', extremely',
    ', intricate,',
]
dangrous_patterns = '[]【】()（）|:：'


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


class FooocusExpansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)
        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)

        self.pipe = pipeline('text-generation',
                             model=self.model,
                             tokenizer=self.tokenizer,
                             device='cpu',
                             torch_dtype=torch.float32)

        print(f'Fooocus Expansion engine loaded.')

    def __call__(self, prompt, seed):
        model_management.load_model_gpu(self.patcher)
        self.pipe.device = self.patcher.load_device
        seed = int(seed)
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]
        response = self.pipe(prompt, max_length=len(prompt) + 256)
        result = response[0]['generated_text'][len(origin):]
        result = safe_str(result)
        result = remove_pattern(result, dangrous_patterns)
        return result
