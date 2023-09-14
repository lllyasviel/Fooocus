import os
import shutil
import safetensors.torch as sf
import torch

virtual_memory_path = './virtual_memory/'
shutil.rmtree(virtual_memory_path, ignore_errors=True)
os.makedirs(virtual_memory_path, exist_ok=True)


def recursive_set(obj, key, value):
    if obj is None:
        return
    if '.' in key:
        k1, k2 = key.split('.', 1)
        recursive_set(getattr(obj, k1, None), k2, value)
    else:
        setattr(obj, key, value)


def force_load_state_dict(model, state_dict):
    for k in list(state_dict.keys()):
        recursive_set(model, k, torch.nn.Parameter(state_dict[k]))
        del state_dict[k]
    return


def move_to_virtual_memory(model):
    model_hash = getattr(model, 'model_hash', None)
    assert isinstance(model_hash, str)
    virtual_memory_filename = getattr(model, 'virtual_memory_filename', None)
    if virtual_memory_filename is not None:
        # Already in virtual memory.
        return
    sd = model.state_dict()
    virtual_memory_device_dict = {k: torch.device(index=v.device.index, type=v.device.type) for k, v in sd.items()}
    virtual_memory_filename = os.path.join(virtual_memory_path, model_hash)
    if not os.path.exists(virtual_memory_filename):
        sf.save_file(sd, virtual_memory_filename)
        print(f'[Virtual Memory System] Tensors written to virtual memory: {virtual_memory_filename}')
    model.virtual_memory_device_dict = virtual_memory_device_dict
    model.virtual_memory_filename = virtual_memory_filename
    model.to('meta')
    print(f'[Virtual Memory System] Tensors released from memory: {virtual_memory_filename}')
    return


def load_from_virtual_memory(model):
    model_hash = getattr(model, 'model_hash', None)
    assert isinstance(model_hash, str)
    virtual_memory_filename = getattr(model, 'virtual_memory_filename', None)
    if virtual_memory_filename is None:
        # Not in virtual memory.
        return
    virtual_memory_device_dict = getattr(model, 'virtual_memory_device_dict', None)
    assert isinstance(virtual_memory_device_dict, dict)
    first_device = list(virtual_memory_device_dict.values())[0].type
    sd = sf.load_file(filename=virtual_memory_filename, device=first_device)
    for k in sd.keys():
        sd[k] = sd[k].to(virtual_memory_device_dict[k])
    force_load_state_dict(model, sd)
    print(f'[Virtual Memory System] Model loaded: {virtual_memory_filename}')
    del model.virtual_memory_device_dict
    del model.virtual_memory_filename
    return


def try_move_to_virtual_memory(model):
    move_to_virtual_memory(model)
