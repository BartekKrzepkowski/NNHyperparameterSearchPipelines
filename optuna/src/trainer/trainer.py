from collections import defaultdict

import torch
from tqdm import tqdm, trange

from src.common.common import LOGGERS_NAME_MAP
from src.common.utils import adjust_evaluators, adjust_evaluators_pre_log, create_paths
from src.models.metrics import prepare_evaluators


class Trainer:
    def __init__(self, model, loaders, criterion, optim, lr_scheduler, device):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = None
        self.epoch_evaluators = None
        self.epoch_denom = None
        self.global_step = None

    def run_exp(self, config):
        """
        Main method of trainer.
        Args:
            epoch_start (int): A number representing the beginning of run
            epoch_end (int): A number representing the end of run
            exp_name (str): Base name of experiment
            config_run_epoch (): ##
            temp (float): CrossEntropy Temperature
            random_seed (int): Seed generator
        """
        self.at_exp_start(config)
        for epoch in tqdm(range(config.epoch_start_at, config.epoch_end_at), desc='run_exp'):
            self.epoch = epoch
            self.model.train()
            self.run_epoch(phase='train', config=config)
            self.model.eval()
            with torch.no_grad():
                self.run_epoch(phase='test', config=config)
            self.logger.close()

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and loggers.
        Args:
            exp_name (str): Base name of experiment
            random_seed (int): seed generator
        Returns:
            save_path (str): Path to save the model
        """
        self.manual_seed(config.random_seed)
        base_path, save_path = create_paths(config.base_path, config.exp_name)
        self.logger = LOGGERS_NAME_MAP[config.logger_name](f'{base_path}/{config.logger_name}', self.device)
        self.base_path = base_path
        self.save_path = save_path

    def run_epoch(self, phase, config):
        self.epoch_evaluators = defaultdict(float)
        self.epoch_denom = 0.0
        running_evaluators = defaultdict(float)
        running_denom = 0.0

        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}', mininterval=30,
                            leave=True, total=loader_size, position=0)
        self.global_step = self.epoch * loader_size
        for i, data in enumerate(progress_bar):
            self.global_step += 1
            x_true, y_true = data
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            loss = self.criterion(y_pred, y_true)
            if 'train' in phase:
                loss /= config.grad_accum_steps
                loss.backward()
                if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == loader_size:
                    self.optim.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optim.zero_grad()
                loss *= config.grad_accum_steps

            # prepare evaluators
            evaluators = prepare_evaluators(y_pred, y_true, loss)

            denom = y_true.size(0)
            running_evaluators = adjust_evaluators(running_evaluators, evaluators, denom, 'running', phase)
            running_denom += denom

            if (i + 1) % (config.grad_accum_steps * config.step_multi) == 0 or (i + 1) == loader_size:
                whether_to_log_running = (i + 1) % (config.grad_accum_steps * config.step_multi) == 0
                self.log(running_evaluators, running_denom, phase, i, progress_bar, whether_to_log_running)
                running_evaluators = defaultdict(float)
                running_denom = 0.0

    def log(self, running_evaluators, running_denom, phase, i, progress_bar, whether_to_log_running):
        self.epoch_evaluators = adjust_evaluators(self.epoch_evaluators, running_evaluators, 1, 'epoch', phase)
        self.epoch_denom += running_denom

        if whether_to_log_running:
            running_evaluators_log = adjust_evaluators_pre_log(running_evaluators, running_denom, round_at=4)
            for tag_evaluator in running_evaluators_log:
                self.logger.log_scalar(tag_evaluator, running_evaluators_log[tag_evaluator], self.global_step)
            progress_bar.set_postfix(running_evaluators_log)

        loader_size = progress_bar.total
        if (i + 1) == loader_size:
            epoch_evaluators_log = adjust_evaluators_pre_log(self.epoch_evaluators, self.epoch_denom, round_at=4)

            for tag_evaluator in epoch_evaluators_log:
                self.logger.log_scalar(tag_evaluator, epoch_evaluators_log[tag_evaluator], self.epoch)

        if self.lr_scheduler is not None and phase == 'train':
            self.logger.log_scalar(f'lr_scheduler', self.lr_scheduler.get_last_lr()[0], self.global_step)

    def manual_seed(self, random_seed):
        """
        Set the environment for reproducibility purposes.
        Args:
            random_seed (int): seed generator
        """
        import random
        import numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(random_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
