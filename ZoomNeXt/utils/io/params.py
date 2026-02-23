import jittor as jt
from jittor import init
from jittor import nn
import os

def save_weight(save_path, model):
    jt.sync_all()
    print(f"Saving weight '{save_path}'")
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = (model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
    jt.save(model_state, save_path)
    print(f"Saved weight '{save_path}' (only contain the net's weight)")

def load_weight(load_path, model, *, strict=True, skip_unmatched_shape=False):
    assert os.path.exists(load_path), load_path
    model_params = model.state_dict()
    for k, v in jt.load(load_path).items():
        if k.endswith('module.'):
            k = k[7:]
        if skip_unmatched_shape and k in model_params and v.shape != model_params[k].shape:
            continue
        model_params[k] = v
    model.load_parameters(model_params)
