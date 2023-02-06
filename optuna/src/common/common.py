import torch

from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger

ACT_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'identity': torch.nn.Identity
}

LOGGERS_NAME_MAP = {
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

LOSS_NAME_MAP = {
    'ce': torch.nn.CrossEntropyLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss
}
