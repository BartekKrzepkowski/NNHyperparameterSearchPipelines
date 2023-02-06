import os
from datetime import datetime

import torch
from torch.nn import functional as F

from src.common.common import OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP


def configure_optimizer(optim_wrapper, model, optim_kwargs, whether_exclude):
    weight_decay = optim_kwargs['weight_decay']
    del optim_kwargs['weight_decay']

    alert_chunks = ['embeddings', 'bn.weight', 'bias']
    no_decay = [pn for pn, p in model.named_parameters() if any(c in pn for c in alert_chunks)] if whether_exclude else []
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in model.named_parameters() if pn not in no_decay and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for pn, p in model.named_parameters() if pn in no_decay and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim_wrapper(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer


def clip_grad_norm(clip_grad_wrapper, model, clip_value):
    clip_grad_wrapper(filter(lambda p: p.requires_grad, model.parameters()), clip_value)


def prepare_optim_and_scheduler(model, optim_name, scheduler_name, optim_params, scheduler_params, whether_exclude):
    optim_wrapper = OPTIMIZER_NAME_MAP[optim_name]
    optim = configure_optimizer(optim_wrapper, model, optim_params, whether_exclude)
    lr_scheduler = SCHEDULER_NAME_MAP[scheduler_name](optim, **scheduler_params)
    return optim, lr_scheduler


def adjust_evaluators(d1, dd2, denom, scope, phase):
    for evaluator_key in dd2:
        eval_key = str(evaluator_key).split('/')
        eval_key = eval_key[0] if len(eval_key) == 1 else '/'.join(eval_key[:-1])
        eval_key = eval_key.split('_')
        eval_key = '_'.join(eval_key[1:]) if eval_key[0] in {'running', 'epoch'} else '_'.join(eval_key)
        d1[f'{scope}_{eval_key}/{phase}'] += dd2[evaluator_key] * denom
    return d1


def adjust_evaluators_pre_log(d1, denom, round_at=4):
    d2 = {}
    for k in d1:
        d2[k] = round(d1[k] / denom, round_at)
    return d2


def update_tensor(a, b):
    c = torch.cat([a, b])
    return c


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def create_paths(base_path, exp_name):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(os.getcwd(), f'{base_path}/{exp_name}/{date}')
    save_path_base = f'{base_path}/checkpoints'
    os.makedirs(save_path_base)
    save_path = lambda step: f'{save_path_base}/model_step_{step}.pth'
    return base_path, save_path


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
