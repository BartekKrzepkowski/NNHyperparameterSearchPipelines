import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

import optuna
from optuna.trial import TrialState

def objective(trial):
    from src.models.models import MLP

    NUM_FEATURES = 32 * 32 * 3
    NUM_CLASSES = 10
    DIMS = [NUM_FEATURES, 512, NUM_CLASSES]

    act_name = trial.suggest_categorical("act", ["relu", "gelu", "tanh"])
    model = MLP(DIMS, act_name).to(device)

    from src.common.common import LOSS_NAME_MAP
    from src.models.losses import ClassificationLoss

    criterion = ClassificationLoss(LOSS_NAME_MAP['ce']())


    from torch.utils.data import DataLoader
    from src.data.datasets import get_cifar10

    train_dataset, _, test_dataset = get_cifar10('data/')

    BATCH_SIZE = 128

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    loaders = {
        'train': train_loader,
        'test': test_loader
    }

    from src.common.common import OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
    GRAD_ACCUM_STEPS = 1
    EPOCHS = 4
    T_max = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optim_params = {'lr': lr, 'weight_decay': weight_decay}
    # scheduler_params = {'T_max': T_max, 'eta_min': 1e-6}

    # optim, lr_scheduler = prepare_optim_and_scheduler(model, 'adamw', 'cosine', optim_params, scheduler_params, whether_exclude=False)

    optim = OPTIMIZER_NAME_MAP['sgd'](filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    lr_scheduler = None #SCHEDULER_NAME_MAP['cosine'](optim, **scheduler_params)

    from src.trainer.trainer_classification import TrainerClassification

    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
    }

    trainer = TrainerClassification(**params_trainer)


    from src.common.utils import AttrDict

    EXP_NAME = 'optuna_lr_wd__mlp_sgd_'

    config = {
        'epoch_start_at': 0,
        'epoch_end_at': EPOCHS,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'save_multi': T_max // 10,
        'log_multi': 100,
        'whether_clip': False,
        'clip_value': 2.0,
        'base_path': 'reports',
        'exp_name': EXP_NAME,
        'logger_name': 'tensorboard',
        'logger_config': {'api_token': "07a2cd842a6d792d578f8e6c0978efeb8dcf7638", 'project': 'early_exit', 'hyperparameters': {}},
        'random_seed': 42,
        'trial': trial,
        'device': device

    }
    config = AttrDict(config)

    acc = trainer.run_exp(config)
    return acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
