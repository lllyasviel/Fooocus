import torch
import gc

from safetensors import safe_open
from comfy import model_management
from comfy.diffusers_convert import textenc_conversion_lst


ALWAYS_USE_VM = None

if ALWAYS_USE_VM is not None:
    print(f'[Virtual Memory System] Forced = {ALWAYS_USE_VM}')

if 'cpu' in model_management.unet_offload_device().type.lower():
    logic_memory = model_management.total_ram
    global_virtual_memory_activated = ALWAYS_USE_VM if ALWAYS_USE_VM is not None else logic_memory < 30000
    print(f'[Virtual Memory System] Logic target is CPU, memory = {logic_memory}')
else:
    logic_memory = model_management.total_vram
    global_virtual_memory_activated = ALWAYS_USE_VM if ALWAYS_USE_VM is not None else logic_memory < 22000
    print(f'[Virtual Memory System] Logic target is GPU, memory = {logic_memory}')


print(f'[Virtual Memory System] Activated = {global_virtual_memory_activated}')


@torch.no_grad()
def recursive_set(obj, key, value):
    if obj is None:
        return
    if '.' in key:
        k1, k2 = key.split('.', 1)
        recursive_set(getattr(obj, k1, None), k2, value)
    else:
        setattr(obj, key, value)


@torch.no_grad()
def recursive_del(obj, key):
    if obj is None:
        return
    if '.' in key:
        k1, k2 = key.split('.', 1)
        recursive_del(getattr(obj, k1, None), k2)
    else:
        delattr(obj, key)


@torch.no_grad()
def force_load_state_dict(model, state_dict):
    for k in list(state_dict.keys()):
        p = torch.nn.Parameter(state_dict[k], requires_grad=False)
        recursive_set(model, k, p)
        del state_dict[k]
    return


@torch.no_grad()
def only_load_safetensors_keys(filename):
    try:
        with safe_open(filename, framework="pt", device='cpu') as f:
            result = list(f.keys())
        assert len(result) > 0
        return result
    except:
        return None


@torch.no_grad()
def move_to_virtual_memory(model, comfy_unload=True):
    if comfy_unload:
        model_management.unload_model()

    virtual_memory_dict = getattr(model, 'virtual_memory_dict', None)
    if isinstance(virtual_memory_dict, dict):
        # Already in virtual memory.
        return

    model_file = getattr(model, 'model_file', None)
    assert isinstance(model_file, dict)

    filename = model_file['filename']
    prefix = model_file['prefix']

    safetensors_keys = only_load_safetensors_keys(filename)

    if safetensors_keys is None:
        print(f'[Virtual Memory System] Error: The Virtual Memory System currently only support safetensors models!')
        return

    sd = model.state_dict()
    original_device = list(sd.values())[0].device.type
    model_file['original_device'] = original_device

    virtual_memory_dict = {}

    for k, v in sd.items():
        current_key = k
        current_flag = None
        if prefix == 'refiner_clip':
            current_key_in_safetensors = k

            for a, b in textenc_conversion_lst:
                current_key_in_safetensors = current_key_in_safetensors.replace(b, a)

            current_key_in_safetensors = current_key_in_safetensors.replace('clip_g.transformer.text_model.encoder.layers.', 'conditioner.embedders.0.model.transformer.resblocks.')
            current_key_in_safetensors = current_key_in_safetensors.replace('clip_g.text_projection', 'conditioner.embedders.0.model.text_projection')
            current_key_in_safetensors = current_key_in_safetensors.replace('clip_g.logit_scale', 'conditioner.embedders.0.model.logit_scale')
            current_key_in_safetensors = current_key_in_safetensors.replace('clip_g.', 'conditioner.embedders.0.model.')

            for e in ["weight", "bias"]:
                for i, k in enumerate(['q', 'k', 'v']):
                    e_flag = f'.{k}_proj.{e}'
                    if current_key_in_safetensors.endswith(e_flag):
                        current_key_in_safetensors = current_key_in_safetensors[:-len(e_flag)] + f'.in_proj_{e}'
                        current_flag = (1280 * i, 1280 * (i + 1))
        else:
            current_key_in_safetensors = prefix + '.' + k
        current_device = torch.device(index=v.device.index, type=v.device.type)
        if current_key_in_safetensors in safetensors_keys:
            virtual_memory_dict[current_key] = (current_key_in_safetensors, current_device, current_flag)
            recursive_del(model, current_key)
        else:
            # print(f'[Virtual Memory System] Missed key: {current_key}')
            pass

    del sd
    gc.collect()
    model_management.soft_empty_cache()

    model.virtual_memory_dict = virtual_memory_dict
    print(f'[Virtual Memory System] {prefix} released from {original_device}: {filename}')
    return


@torch.no_grad()
def load_from_virtual_memory(model):
    virtual_memory_dict = getattr(model, 'virtual_memory_dict', None)
    if not isinstance(virtual_memory_dict, dict):
        # Not in virtual memory.
        return

    model_file = getattr(model, 'model_file', None)
    assert isinstance(model_file, dict)

    filename = model_file['filename']
    prefix = model_file['prefix']
    original_device = model_file['original_device']

    with safe_open(filename, framework="pt", device=original_device) as f:
        for current_key, (current_key_in_safetensors, current_device, current_flag) in virtual_memory_dict.items():
            tensor = f.get_tensor(current_key_in_safetensors).to(current_device)
            if isinstance(current_flag, tuple) and len(current_flag) == 2:
                a, b = current_flag
                tensor = tensor[a:b]
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            recursive_set(model, current_key, parameter)

    print(f'[Virtual Memory System] {prefix} loaded to {original_device}: {filename}')
    del model.virtual_memory_dict
    return


@torch.no_grad()
def try_move_to_virtual_memory(model, comfy_unload=True):
    if not global_virtual_memory_activated:
        return

    import modules.default_pipeline as pipeline

    if pipeline.xl_refiner is None:
        # If users do not use refiner, no need to use this.
        return

    move_to_virtual_memory(model, comfy_unload)
