import torch
import comfy.model_base
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.samplers
import modules.anisotropic as anisotropic

from comfy.samplers import model_management, lcm, math
from comfy.ldm.modules.diffusionmodules.openaimodel import timestep_embedding, forward_timestep_embed


sharpness = 2.0


def sampling_function_patched(model_function, x, timestep, uncond, cond, cond_scale, cond_concat=None, model_options={},
                      seed=None):
    def get_area_and_mult(cond, x_in, cond_concat_in, timestep_in):
        area = (x_in.shape[2], x_in.shape[3], 0, 0)
        strength = 1.0
        if 'timestep_start' in cond[1]:
            timestep_start = cond[1]['timestep_start']
            if timestep_in[0] > timestep_start:
                return None
        if 'timestep_end' in cond[1]:
            timestep_end = cond[1]['timestep_end']
            if timestep_in[0] < timestep_end:
                return None
        if 'area' in cond[1]:
            area = cond[1]['area']
        if 'strength' in cond[1]:
            strength = cond[1]['strength']

        adm_cond = None
        if 'adm_encoded' in cond[1]:
            adm_cond = cond[1]['adm_encoded']

        input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        if 'mask' in cond[1]:
            # Scale the mask to the size of the input
            # The mask should have been resized as we began the sampling process
            mask_strength = 1.0
            if "mask_strength" in cond[1]:
                mask_strength = cond[1]["mask_strength"]
            mask = cond[1]['mask']
            assert (mask.shape[1] == x_in.shape[2])
            assert (mask.shape[2] == x_in.shape[3])
            mask = mask[:, area[2]:area[0] + area[2], area[3]:area[1] + area[3]] * mask_strength
            mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
        else:
            mask = torch.ones_like(input_x)
        mult = mask * strength

        if 'mask' not in cond[1]:
            rr = 8
            if area[2] != 0:
                for t in range(rr):
                    mult[:, :, t:1 + t, :] *= ((1.0 / rr) * (t + 1))
            if (area[0] + area[2]) < x_in.shape[2]:
                for t in range(rr):
                    mult[:, :, area[0] - 1 - t:area[0] - t, :] *= ((1.0 / rr) * (t + 1))
            if area[3] != 0:
                for t in range(rr):
                    mult[:, :, :, t:1 + t] *= ((1.0 / rr) * (t + 1))
            if (area[1] + area[3]) < x_in.shape[3]:
                for t in range(rr):
                    mult[:, :, :, area[1] - 1 - t:area[1] - t] *= ((1.0 / rr) * (t + 1))

        conditionning = {}
        conditionning['c_crossattn'] = cond[0]
        if cond_concat_in is not None and len(cond_concat_in) > 0:
            cropped = []
            for x in cond_concat_in:
                cr = x[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
                cropped.append(cr)
            conditionning['c_concat'] = torch.cat(cropped, dim=1)

        if adm_cond is not None:
            conditionning['c_adm'] = adm_cond

        control = None
        if 'control' in cond[1]:
            control = cond[1]['control']

        patches = None
        if 'gligen' in cond[1]:
            gligen = cond[1]['gligen']
            patches = {}
            gligen_type = gligen[0]
            gligen_model = gligen[1]
            if gligen_type == "position":
                gligen_patch = gligen_model.set_position(input_x.shape, gligen[2], input_x.device)
            else:
                gligen_patch = gligen_model.set_empty(input_x.shape, input_x.device)

            patches['middle_patch'] = [gligen_patch]

        return (input_x, mult, conditionning, area, control, patches)

    def cond_equal_size(c1, c2):
        if c1 is c2:
            return True
        if c1.keys() != c2.keys():
            return False
        if 'c_crossattn' in c1:
            s1 = c1['c_crossattn'].shape
            s2 = c2['c_crossattn'].shape
            if s1 != s2:
                if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                    return False

                mult_min = lcm(s1[1], s2[1])
                diff = mult_min // min(s1[1], s2[1])
                if diff > 4:  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                    return False
        if 'c_concat' in c1:
            if c1['c_concat'].shape != c2['c_concat'].shape:
                return False
        if 'c_adm' in c1:
            if c1['c_adm'].shape != c2['c_adm'].shape:
                return False
        return True

    def can_concat_cond(c1, c2):
        if c1[0].shape != c2[0].shape:
            return False

        # control
        if (c1[4] is None) != (c2[4] is None):
            return False
        if c1[4] is not None:
            if c1[4] is not c2[4]:
                return False

        # patches
        if (c1[5] is None) != (c2[5] is None):
            return False
        if (c1[5] is not None):
            if c1[5] is not c2[5]:
                return False

        return cond_equal_size(c1[2], c2[2])

    def cond_cat(c_list):
        c_crossattn = []
        c_concat = []
        c_adm = []
        crossattn_max_len = 0
        for x in c_list:
            if 'c_crossattn' in x:
                c = x['c_crossattn']
                if crossattn_max_len == 0:
                    crossattn_max_len = c.shape[1]
                else:
                    crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
                c_crossattn.append(c)
            if 'c_concat' in x:
                c_concat.append(x['c_concat'])
            if 'c_adm' in x:
                c_adm.append(x['c_adm'])
        out = {}
        c_crossattn_out = []
        for c in c_crossattn:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)  # padding with repeat doesn't change result
            c_crossattn_out.append(c)

        if len(c_crossattn_out) > 0:
            out['c_crossattn'] = [torch.cat(c_crossattn_out)]
        if len(c_concat) > 0:
            out['c_concat'] = [torch.cat(c_concat)]
        if len(c_adm) > 0:
            out['c_adm'] = torch.cat(c_adm)
        return out

    def calc_cond_uncond_batch(model_function, cond, uncond, x_in, timestep, max_total_area, cond_concat_in,
                               model_options):
        out_cond = torch.zeros_like(x_in)
        out_count = torch.ones_like(x_in) / 100000.0

        out_uncond = torch.zeros_like(x_in)
        out_uncond_count = torch.ones_like(x_in) / 100000.0

        COND = 0
        UNCOND = 1

        to_run = []
        for x in cond:
            p = get_area_and_mult(x, x_in, cond_concat_in, timestep)
            if p is None:
                continue

            to_run += [(p, COND)]
        if uncond is not None:
            for x in uncond:
                p = get_area_and_mult(x, x_in, cond_concat_in, timestep)
                if p is None:
                    continue

                to_run += [(p, UNCOND)]

        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp) // i]
                if (len(batch_amount) * first_shape[0] * first_shape[2] * first_shape[3] < max_total_area):
                    to_batch = batch_amount
                    break

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x += [p[0]]
                mult += [p[1]]
                c += [p[2]]
                area += [p[3]]
                cond_or_uncond += [o[1]]
                control = p[4]
                patches = p[5]

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

            transformer_options = {}
            if 'transformer_options' in model_options:
                transformer_options = model_options['transformer_options'].copy()

            if patches is not None:
                if "patches" in transformer_options:
                    cur_patches = transformer_options["patches"].copy()
                    for p in patches:
                        if p in cur_patches:
                            cur_patches[p] = cur_patches[p] + patches[p]
                        else:
                            cur_patches[p] = patches[p]
                else:
                    transformer_options["patches"] = patches

            c['transformer_options'] = transformer_options

            transformer_options['uc_mask'] = torch.Tensor(cond_or_uncond).to(input_x).float()[:, None, None, None]

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model_function,
                                                                 {"input": input_x, "timestep": timestep_, "c": c,
                                                                  "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
            else:
                output = model_function(input_x, timestep_, **c).chunk(batch_chunks)
            del input_x

            model_management.throw_exception_if_processing_interrupted()

            for o in range(batch_chunks):
                if cond_or_uncond[o] == COND:
                    out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[
                                                                                                                  o] * \
                                                                                                              mult[o]
                    out_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
                else:
                    out_uncond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[
                                                                                                                    o] * \
                                                                                                                mult[o]
                    out_uncond_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += \
                    mult[o]
            del mult

        out_cond /= out_count
        del out_count
        out_uncond /= out_uncond_count
        del out_uncond_count

        return out_cond, out_uncond

    max_total_area = model_management.maximum_batch_area()
    if math.isclose(cond_scale, 1.0):
        uncond = None

    cond, uncond = calc_cond_uncond_batch(model_function, cond, uncond, x, timestep, max_total_area, cond_concat,
                                          model_options)
    if "sampler_cfg_function" in model_options:
        args = {"cond": cond, "uncond": uncond, "cond_scale": cond_scale, "timestep": timestep}
        return model_options["sampler_cfg_function"](args)
    else:
        return uncond + (cond - uncond) * cond_scale


def unet_forward_patched(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    uc_mask = transformer_options['uc_mask']
    transformer_options["original_shape"] = list(x.shape)
    transformer_options["current_index"] = 0

    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options)
        if control is not None and 'input' in control and len(control['input']) > 0:
            ctrl = control['input'].pop()
            if ctrl is not None:
                h += ctrl
        hs.append(h)
    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
    if control is not None and 'middle' in control and len(control['middle']) > 0:
        h += control['middle'].pop()

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        if control is not None and 'output' in control and len(control['output']) > 0:
            ctrl = control['output'].pop()
            if ctrl is not None:
                hsp += ctrl

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
    h = h.type(x.dtype)
    x0 = self.out(h)

    alpha = 1.0 - (timesteps / 999.0)[:, None, None, None].clone()
    alpha *= 0.001 * sharpness
    degraded_x0 = anisotropic.bilateral_blur(x0) * alpha + x0 * (1.0 - alpha)

    x0 = x0 * uc_mask + degraded_x0 * (1.0 - uc_mask)

    return x0


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = kwargs["pooled_output"]
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    crop_w = kwargs.get("crop_w", 0)
    crop_h = kwargs.get("crop_h", 0)
    target_width = kwargs.get("target_width", width)
    target_height = kwargs.get("target_height", height)

    if kwargs.get("prompt_type", "") == "negative":
        width *= 0.8
        height *= 0.8
    elif kwargs.get("prompt_type", "") == "positive":
        width *= 1.5
        height *= 1.5

    out = []
    out.append(self.embedder(torch.Tensor([height])))
    out.append(self.embedder(torch.Tensor([width])))
    out.append(self.embedder(torch.Tensor([crop_h])))
    out.append(self.embedder(torch.Tensor([crop_w])))
    out.append(self.embedder(torch.Tensor([target_height])))
    out.append(self.embedder(torch.Tensor([target_width])))
    flat = torch.flatten(torch.cat(out))[None, ]
    return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def patch_all():
    comfy.samplers.sampling_function = sampling_function_patched
    comfy.model_base.SDXL.encode_adm = sdxl_encode_adm_patched
    comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = unet_forward_patched
