import os
import logging
from datetime import datetime

from tools.load_data import get_device, load_data
from experiment import run_experiments
from optim.utils import MetaData
from tools.load_imagenet_parquet import load_ImageNet100, load_ImageNetTiny
from tools.util_experiments import initialise_experiments_by_mask, load_resnet, get_extra_params,\
    mask_netline, mask_dz, mask_lookahead, mask_leap, mask_sgd
from tools.util_optimisers import clone_resnet18v2, clone_resnet34v2

device = get_device()

# =====================
import sys 
from pathlib import Path

all_exp = [mask_netline, mask_dz, mask_lookahead, mask_leap, mask_sgd]

if __name__ == '__main__' and Path(sys.argv[0]).stem not in {"ipython", "ipython3", "ipykernel_launcher"}:
    run_test = int(sys.argv[1])

if 'run_test' in locals():
    os.makedirs(".data_experiments", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

lr1 = 5e-5
lr_max = 0.02
val_linearity_coeff = 0.1 #0.2 #0.05
val_momentum = 0.9
val_weight_decay = 5e-3
val_sigma_coeff = 0.08 #0.16 #0.04
la_steps=5
la_alpha=0.8
val_decoupled_wd=False

#Exp100 RESNET18/CIFAR10 with lookahead on
if 'run_test' in locals() and run_test == 100:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp100_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R18D10', net_num = manual_seed).to(device)

    print(f"Experiment_100 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    for curr_exp in all_exp:
        if mask_exp & curr_exp > 0:
            exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                                val_linearity_coeff, la_alpha, val_sigma_coeff, curr_exp, fn_clone = clone_resnet18v2,\
                                                decoupled_wd=val_decoupled_wd)
            run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp101 RESNET18/CIFAR10 with lookahead off
if 'run_test' in locals() and run_test == 101:
    la_alpha = 1.0

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp101_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R18D10', net_num = manual_seed).to(device)

    print(f"Experiment_101 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    for curr_exp in all_exp:
        if mask_exp & curr_exp > 0:
            exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                                val_linearity_coeff, la_alpha, val_sigma_coeff, curr_exp, fn_clone = clone_resnet18v2,\
                                                decoupled_wd=val_decoupled_wd)
            run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp110 RESNET18/CIFAR100 with lookahead on
if 'run_test' in locals() and run_test == 110:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp110_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R18D100', net_num = manual_seed).to(device)

    print(f"Experiment_110 RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    for curr_exp in all_exp:
        if mask_exp & curr_exp > 0:
            exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                                val_linearity_coeff, la_alpha, val_sigma_coeff, curr_exp, fn_clone = clone_resnet18v2,\
                                                decoupled_wd=val_decoupled_wd)
            run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp111 RESNET18/CIFAR100 with lookahead off
if 'run_test' in locals() and run_test == 111:
    la_alpha = 1.0

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp111_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R18D100', net_num = manual_seed).to(device)

    print(f"Experiment_111 RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    for curr_exp in all_exp:
        if mask_exp & curr_exp > 0:
            exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                                val_linearity_coeff, la_alpha, val_sigma_coeff, curr_exp, fn_clone = clone_resnet18v2,\
                                                decoupled_wd=val_decoupled_wd)
            run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
