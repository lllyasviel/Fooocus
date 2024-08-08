# Consistent with Kohya/A1111 to reduce differences between model training and inference.

import json
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
import ldm_patched.modules.samplers
import ldm_patched.modules.sd
import ldm_patched.modules.sd1_clip
import ldm_patched.modules.clip_vision
import ldm_patched.modules.ops as ops

from modules.ops import use_patched_ops
from transformers import CLIPTextModel, CLIPTextConfig, modeling_utils, CLIPVisionConfig, CLIPVisionModelWithProjection


def patched_encode_token_weights(self, token_weight_pairs):
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


def patched_SDClipModel__init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=ldm_patched.modules.clip_model.CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, return_projected_pooled=True):  # clip-vit-base-patch32
    torch.nn.Module.__init__(self)
    assert layer in self.LAYERS

    if textmodel_json_config is None:
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(ldm_patched.modules.sd1_clip.__file__)), "sd1_clip_config.json")

    with open(textmodel_json_config) as f:
        config = json.load(f)

    self.transformer = model_class(config, dtype, device, ldm_patched.modules.ops.manual_cast)
    self.num_layers = self.transformer.num_layers
    self.transformer.text_model.embeddings.to(torch.float32)

    self.max_length = max_length
    if freeze:
        self.freeze()
    self.layer = layer
    self.layer_idx = None
    self.special_tokens = special_tokens

    # TODO check if necessary
    self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
    self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
    self.enable_attention_masks = enable_attention_masks

    self.layer_norm_hidden_state = layer_norm_hidden_state
    self.return_projected_pooled = return_projected_pooled

    if layer == "hidden":
        assert layer_idx is not None
        assert abs(layer_idx) < self.num_layers
        self.set_clip_options({"layer": layer_idx})
    self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)


def patched_SDClipModel_forward(self, tokens):
    backup_embeds = self.transformer.get_input_embeddings()
    device = backup_embeds.weight.device
    tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
    tokens = torch.LongTensor(tokens).to(device)

    attention_mask = None
    if self.enable_attention_masks:
        attention_mask = torch.zeros_like(tokens)
        max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if tokens[x, y] == max_token:
                    break

    outputs = self.transformer(tokens, attention_mask, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
    self.transformer.set_input_embeddings(backup_embeds)

    if self.layer == "last":
        z = outputs.last_hidden_state
    elif self.layer == "pooled":
        z = outputs.pooler_output[:, None, :]
    else:
        z = outputs[1]

    pooled_output = None
    if len(outputs) >= 3:
        if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
            pooled_output = outputs[3].float()
        elif outputs[2] is not None:
            pooled_output = outputs[2].float()

    return z.float(), pooled_output


def patched_ClipVisionModel__init__(self, json_config):
    config = CLIPVisionConfig.from_json_file(json_config)

    self.load_device = ldm_patched.modules.model_management.text_encoder_device()
    self.offload_device = ldm_patched.modules.model_management.text_encoder_offload_device()

    if ldm_patched.modules.model_management.should_use_fp16(self.load_device, prioritize_performance=False):
        self.dtype = torch.float16
    else:
        self.dtype = torch.float32

    with use_patched_ops(ops.manual_cast):
        with modeling_utils.no_init_weights():
            self.model = CLIPVisionModelWithProjection(config)

    self.model.to(self.dtype)
    self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(
        self.model,
        load_device=self.load_device,
        offload_device=self.offload_device
    )


def patched_ClipVisionModel_encode_image(self, image):
    ldm_patched.modules.model_management.load_model_gpu(self.patcher)
    pixel_values = ldm_patched.modules.clip_vision.clip_preprocess(image.to(self.load_device))
    outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

    for k in outputs:
        t = outputs[k]
        if t is not None:
            if k == 'hidden_states':
                outputs["penultimate_hidden_states"] = t[-2].to(ldm_patched.modules.model_management.intermediate_device())
                outputs["hidden_states"] = None
            else:
                outputs[k] = t.to(ldm_patched.modules.model_management.intermediate_device())

    return outputs


def patch_all_clip():
    ldm_patched.modules.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = patched_encode_token_weights
    ldm_patched.modules.sd1_clip.SDClipModel.__init__ = patched_SDClipModel__init__
    ldm_patched.modules.sd1_clip.SDClipModel.forward = patched_SDClipModel_forward
    ldm_patched.modules.clip_vision.ClipVisionModel.__init__ = patched_ClipVisionModel__init__
    ldm_patched.modules.clip_vision.ClipVisionModel.encode_image = patched_ClipVisionModel_encode_image
    return
