import os
import logging
from datetime import datetime

from tools.load_data import get_device, load_data
from experiment import run_experiments
from optim.utils import MetaData
from tools.load_imagenet_parquet import load_ImageNet100, load_ImageNetTiny
from tools.util_experiments import initialise_experiments_by_mask, load_resnet
from tools.util_optimisers import clone_resnet18v2, clone_resnet34v2

device = get_device()

# =====================
import sys 
from pathlib import Path


def get_job_nr(args):
    if len(args) >= 3:
        return args[2]
    else:
        return '0'

def get_epochs_per_exp(args):
    if len(args) >= 4:
        return int(args[3])
    else:
        return 50

def get_mask(args):
    if len(args) >= 5:
        return int(args[4])
    else:
        return 31

def get_manual_seed(args):
    if len(args) >= 6:
        return int(args[5])
    else:
        return -1

def get_extra_params(args):
    return get_job_nr(args), get_epochs_per_exp(args), get_mask(args), get_manual_seed(args)

if __name__ == '__main__' and Path(sys.argv[0]).stem not in {"ipython", "ipython3", "ipykernel_launcher"}:
    run_test = int(sys.argv[1])

if 'run_test' in locals():
    os.makedirs(".data_experiments", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

lr1 = 5e-5
lr_max = 0.02
val_linearity_coeff = 0.05
val_momentum = 0.9
val_weight_decay = 5e-3
val_sigma_coeff = 0.04
la_steps=5
la_alpha=0.8

#Exp100 RESNET18/CIFAR10 with lookahead on
if 'run_test' in locals() and run_test == 100:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp100_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R18D10', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_100 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp101 RESNET18/CIFAR10 with lookahead off
if 'run_test' in locals() and run_test == 101:
    la_alpha, val_linearity_coeff, val_sigma_coeff = 1.0, 0.1, 0.08

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp101_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R18D10', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_101 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp110 RESNET18/CIFAR100 with lookahead on
if 'run_test' in locals() and run_test == 110:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp111_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R18D100', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_110 RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp111 RESNET18/CIFAR100 with lookahead off
if 'run_test' in locals() and run_test == 111:
    la_alpha, linearity_coeff_111 = 1.0, 0.1

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp111_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R18D100', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          linearity_coeff_111, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_111 RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp120 RESNET34/CIFAR10 with lookahead on
if 'run_test' in locals() and run_test == 120:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp120_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R34D10', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet34v2)

    print(f"Experiment_120 RESNET34/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp121 RESNET34/CIFAR10 with lookahead off
if 'run_test' in locals() and run_test == 121:
    la_alpha, val_linearity_coeff = 1.0, 0.1

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp121_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('CIFAR10', manual_seed = manual_seed)
    model = load_resnet('R34D10', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet34v2)

    print(f"Experiment_121 RESNET34/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp130 RESNET34/CIFAR100 with lookahead on
if 'run_test' in locals() and run_test == 130:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp130_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R34D100', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet34v2)

    print(f"Experiment_130 RESNET34/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp131 RESNET34/CIFAR100 with lookahead off
if 'run_test' in locals() and run_test == 131:
    la_alpha, linearity_coeff_131 = 1.0, 0.1

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp131_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_data('CIFAR100', manual_seed = manual_seed)
    model = load_resnet('R34D100', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          linearity_coeff_131, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet34v2)

    print(f"Experiment_131 RESNET34/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp140 RESNET18/SVHN with lookahead on
if 'run_test' in locals() and run_test == 140:

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp140_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    train_dl, test_dl = load_data('SVHN', manual_seed = manual_seed)
    model = load_resnet('R18D10', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          val_linearity_coeff, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_140 RESNET18/SVHN/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp150 RESNET18/ImageNet100 with lookahead off
if 'run_test' in locals() and run_test == 150:
    la_alpha, linearity_coeff_150 = 1.0, 0.1

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp150_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    train_dl, test_dl = load_ImageNet100(manual_seed = manual_seed)
    model = load_resnet('R18D100', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          linearity_coeff_150, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_150 RESNET18/ImageNet100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp151 RESNET18/TinyImageNet with lookahead off
if 'run_test' in locals() and run_test == 151:
    la_alpha, linearity_coeff_151 = 1.0, 0.1

    job_nr_str, EPOCHS_PER_EXPERIMENT, mask_exp, manual_seed = get_extra_params(sys.argv)
    logging.basicConfig(filename=f"logs/exp151_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=200, device=device)
    train_dl, test_dl = load_ImageNetTiny(manual_seed = manual_seed)
    model = load_resnet('R18D200', net_num = manual_seed).to(device)

    exps = initialise_experiments_by_mask(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                          linearity_coeff_151, la_alpha, val_sigma_coeff, mask_exp, fn_clone = clone_resnet18v2)

    print(f"Experiment_151 RESNET18/TinyImageNet/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} with manual_seed={manual_seed} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, exps, num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
