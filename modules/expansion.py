import torch

import comfy.model_management as model_management

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from modules.path import fooocus_expansion_path
from comfy.model_patcher import ModelPatcher


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
        self.model.eval()

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()

        # MPS hack
        if model_management.is_device_mps(load_device):
            load_device = torch.device('cpu')
            offload_device = torch.device('cpu')

        use_fp16 = model_management.should_use_fp16(device=load_device)

        if use_fp16:
            self.model.half()

        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)
        print(f'Fooocus Expansion engine loaded for {load_device}, use_fp16 = {use_fp16}.')

    def __call__(self, prompt, seed):
        if self.patcher.current_device != self.patcher.load_device:
            print('Fooocus Expansion loaded by itself.')
            model_management.load_model_gpu(self.patcher)

        seed = int(seed)
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       num_beams=1,
                                       max_new_tokens=256,
                                       do_sample=True)

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = response[0][len(origin):]
        result = safe_str(result)
        result = remove_pattern(result, dangrous_patterns)
        return result
