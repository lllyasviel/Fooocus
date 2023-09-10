import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from modules.path import fooocus_expansion_path


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
        prompt = str(prompt).rstrip('\n')
        seed = int(seed)
        set_seed(seed)
        response = self.pipe(prompt, max_length=len(prompt) + 256)
        result = response[0]['generated_text'].rstrip('\n')
        return result
