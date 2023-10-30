import torch
import math
import fcbh.model_management as model_management

from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from modules.path import fooocus_expansion_path
from fcbh.model_patcher import ModelPatcher

# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32


fooocus_magic_split = [
    ', extremely',
    ', intricate,',
]
dangrous_patterns = '[]【】()（）|:：'

black_list = ['art', 'digital', 'paint', 'painting', 'painted', 'drawing', 'draw', 'drawn',
              'concept', 'illustration', 'illustrated', 'illustrate',
              'face', 'eye', 'eyes', 'hand', 'hands', 'head', 'heads', 'leg', 'legs', 'arm', 'arms',
              'shoulder', 'shoulders', 'body', 'facial', 'skin', 'character', 'human', 'portrait', 'cloth'
              'monster', 'artistic', 'oil', 'brush',
              'artwork', 'artworks',
              'skeletal', 'skeleton', 'a', 'the', 'background']

black_list += ['Ġ' + k for k in black_list]
black_list += [k.upper() for k in black_list]
black_list += [k.capitalize() for k in black_list]
black_list += ['Ġ' + k.upper() for k in black_list]
black_list += ['Ġ' + k.capitalize() for k in black_list]


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
        self.vocab = self.tokenizer.vocab
        self.logits_bias = torch.zeros((1, len(self.vocab)), dtype=torch.float32)
        self.logits_bias[0, self.tokenizer.eos_token_id] = - 16.0
        self.logits_bias[0, 198] = - 1024.0  # test_198 = self.tokenizer('\n', return_tensors="pt")
        for k, v in self.vocab.items():
            if k in black_list:
                self.logits_bias[0, v] = - 1024.0

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

    def logits_processor(self, input_ids, scores):
        self.logits_bias = self.logits_bias.to(scores)
        return scores + self.logits_bias

    def __call__(self, prompt, seed):
        if prompt == '':
            return ''

        if self.patcher.current_device != self.patcher.load_device:
            print('Fooocus Expansion loaded by itself.')
            model_management.load_model_gpu(self.patcher)

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        logits_processor = LogitsProcessorList([self.logits_processor])

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       num_beams=1,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=logits_processor)

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = response[0][len(origin):]
        result = safe_str(result)
        result = remove_pattern(result, dangrous_patterns)
        return result
