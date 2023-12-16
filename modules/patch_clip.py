# Consistent with Kohya/A1111 to reduce differences between model training and inference.

import os
import torch
import ldm_patched.controlnet.cldm
import ldm_patched.k_diffusion.sampling
import ldm_patched.ldm.modules.attention
import ldm_patched.ldm.modules.diffusionmodules.model
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.modules.args_parser
import ldm_patched.modules.model_base
import ldm_patched.modules.model_management
import ldm_patched.modules.model_patcher
import ldm_patched.modules.ops
import ldm_patched.modules.samplers
import ldm_patched.modules.sd
import ldm_patched.modules.sd1_clip
import ldm_patched.modules.clip_vision
import ldm_patched.modules.model_management as model_management
import contextlib

from transformers import CLIPTextModel, CLIPTextConfig, modeling_utils, CLIPVisionConfig, CLIPVisionModelWithProjection


@contextlib.contextmanager
def use_disable_weight_init_linear_ops(device=None, dtype=None):
    old_torch_nn_linear = torch.nn.Linear
    force_device = device
    force_dtype = dtype

    def linear_with_dtype(in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        if force_device is not None:
            device = force_device
        if force_dtype is not None:
            dtype = force_dtype
        return ldm_patched.modules.ops.disable_weight_init.Linear(in_features, out_features, bias=bias, device=device,
                                                                  dtype=dtype)

    torch.nn.Linear = linear_with_dtype
    try:
        yield
    finally:
        torch.nn.Linear = old_torch_nn_linear
    return


def encode_token_weights_fooocus(self, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        to_encode.append(ldm_patched.modules.sd1_clip.gen_empty_tokens(self.special_tokens, max_token_len))

    out, pooled = self.encode(to_encode)
    if pooled is not None:
        first_pooled = pooled[0:1].to(ldm_patched.modules.model_management.intermediate_device())
    else:
        first_pooled = pooled

    output = []
    for k in range(0, sections):
        z = out[k:k + 1]
        if has_weights:
            original_mean = z.mean()
            z_empty = out[-1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            new_mean = z.mean()
            z = z * (original_mean / new_mean)
        output.append(z)

    if len(output) == 0:
        return out[-1:].to(ldm_patched.modules.model_management.intermediate_device()), first_pooled
    return torch.cat(output, dim=-2).to(ldm_patched.modules.model_management.intermediate_device()), first_pooled


class SDClipModelFooocus(torch.nn.Module, ldm_patched.modules.sd1_clip.ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=ldm_patched.modules.clip_model.CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(ldm_patched.modules.sd1_clip.__file__)), "sd1_clip_config.json")

        config = CLIPTextConfig.from_json_file(textmodel_json_config)

        self.num_layers = config.num_hidden_layers
        with use_disable_weight_init_linear_ops(device, dtype):
            with modeling_utils.no_init_weights():
                self.transformer = CLIPTextModel(config)

        self.inner_name = "text_model"
        if dtype is not None:
            self.transformer.to(dtype)
            inner_model = getattr(self.transformer, self.inner_name)
            if hasattr(inner_model, "embeddings"):
                inner_model.embeddings.to(torch.float32)
            else:
                self.transformer.set_input_embeddings(self.transformer.get_input_embeddings().to(torch.float32))

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = False

        self.layer_norm_hidden_state = layer_norm_hidden_state
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) <= self.num_layers
            self.clip_layer(layer_idx)
        self.layer_default = (self.layer, self.layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) >= self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:  # EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored",
                              y.shape[0], current_embeds.weight.shape[1])
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1],
                                               device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        if getattr(self.transformer, self.inner_name).final_layer_norm.weight.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, dtype: contextlib.nullcontext(a)

        with precision_scope(model_management.get_autocast_device(device), dtype=torch.float32):
            attention_mask = None
            if self.enable_attention_masks:
                attention_mask = torch.zeros_like(tokens)
                max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
                for x in range(attention_mask.shape[0]):
                    for y in range(attention_mask.shape[1]):
                        attention_mask[x, y] = 1
                        if tokens[x, y] == max_token:
                            break

            outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask,
                                       output_hidden_states=self.layer == "hidden")
            self.transformer.set_input_embeddings(backup_embeds)

            if self.layer == "last":
                z = outputs.last_hidden_state
            elif self.layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
                if self.layer_norm_hidden_state:
                    z = getattr(self.transformer, self.inner_name).final_layer_norm(z)

            if hasattr(outputs, "pooler_output"):
                pooled_output = outputs.pooler_output.float()
            else:
                pooled_output = None

            if self.text_projection is not None and pooled_output is not None:
                pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()
        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        if "text_projection" in sd:
            self.text_projection[:] = sd.pop("text_projection")
        if "text_projection.weight" in sd:
            self.text_projection[:] = sd.pop("text_projection.weight").transpose(0, 1)
        return self.transformer.load_state_dict(sd, strict=False)


class ClipVisionModelFooocus:
    def __init__(self, json_config):
        config = CLIPVisionConfig.from_json_file(json_config)
        self.load_device = ldm_patched.modules.model_management.text_encoder_device()
        offload_device = ldm_patched.modules.model_management.text_encoder_offload_device()
        self.dtype = torch.float32
        if ldm_patched.modules.model_management.should_use_fp16(self.load_device, prioritize_performance=False):
            self.dtype = torch.float16

        with use_disable_weight_init_linear_ops(offload_device, self.dtype):
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.dtype)

        self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def encode_image(self, image):
        raise NotImplementedError('wrong clip vision call!')


def patch_all_clip():
    ldm_patched.modules.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = encode_token_weights_fooocus
    ldm_patched.modules.sd1_clip.SDClipModel = SDClipModelFooocus
    ldm_patched.modules.clip_vision.ClipVisionModel = ClipVisionModelFooocus
    return
