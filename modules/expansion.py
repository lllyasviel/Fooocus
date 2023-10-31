import os
import torch
import math
import fcbh.model_management as model_management

from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from modules.path import fooocus_expansion_path
from fcbh.model_patcher import ModelPatcher

# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32


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

        positive_words = open(os.path.join(fooocus_expansion_path, 'positive.txt'), encoding='utf-8').read()
        positive_words = positive_words.lower().replace(' ', '').replace('\n', '').split(',')

        # print(', '.join(sorted(list(set(positive_words)))))

        # t198 = self.tokenizer('\n', return_tensors="np")
        # t11 = self.tokenizer(',', return_tensors="np")
        # positive_ids = [11, 198, self.tokenizer.eos_token_id]
        positive_ids = [11]

        self.bad_words_ids = []
        for k, v in self.tokenizer.vocab.items():
            if k.replace('Ġ', '') not in positive_words and v not in positive_ids:
                self.bad_words_ids.append([v])

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
        if prompt == '':
            return ''

        if self.patcher.current_device != self.patcher.load_device:
            print('Fooocus Expansion loaded by itself.')
            model_management.load_model_gpu(self.patcher)

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       bad_words_ids=self.bad_words_ids)

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result
