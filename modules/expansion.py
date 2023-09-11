import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from modules.path import fooocus_expansion_path


fooocus_magic_split = [
    ', extremely',
    ', trending',
    ', intricate',
    # '. The',
]
dangrous_patterns = '[]【】()（）|:：'


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.rstrip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


class FooocusExpansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)
        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)
        self.pipe = pipeline('text-generation',
                             model=self.model,
                             tokenizer=self.tokenizer,
                             device='cpu',
                             torch_dtype=torch.float32)
        print('Fooocus Expansion engine loaded.')

    def __call__(self, prompt, seed):
        seed = int(seed)
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]
        response = self.pipe(prompt, max_length=len(prompt) + 256)
        result = response[0]['generated_text'][len(origin):]
        result = safe_str(result)
        result = remove_pattern(result, dangrous_patterns)
        return result
