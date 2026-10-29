import os
import logging
from datetime import datetime

from tools.load_data import get_device, load_data
from experiment import run_experiments
from optim.utils import MetaData
from nets.cnn import _make_resnet18v2, _make_resnet34v2
from tools.load_imagenet_parquet import load_ImageNet100, load_ImageNetTiny
from tools.experiments_util import initialise_netline_experiment, initialise_dz_experiment, initialise_lookahead_experiment,\
    initialise_sgd_experiment, initialise_leap_experiment

device = get_device()

# =====================
import sys 
from pathlib import Path

def clone_resnet18v2(model, meta):
    mdl = _make_resnet18v2(meta.output_dim).to(device)
    mdl.load_state_dict(model.state_dict())
    return mdl

def clone_resnet34v2(model, meta):
    mdl = _make_resnet34v2(meta.output_dim).to(device)
    mdl.load_state_dict(model.state_dict())
    return mdl

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
val_sigma_coeff = 0.01
la_steps=5
la_alpha=0.8

def format_params():
    return f"lr1={lr1}, lr_max={lr_max}, momentum={val_momentum}, la_steps={la_steps}, la_alpha={la_alpha}"

#Exp100 RESNET18/CIFAR10 with lookahead on
if 'run_test' in locals() and run_test == 100:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp100_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay, val_sigma_coeff)

    print(f"Experiment_100 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp101 RESNET18/CIFAR10 with lookahead off
if 'run_test' in locals() and run_test == 101:
    la_alpha, val_linearity_coeff = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp101_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay, val_sigma_coeff)

    print(f"Experiment_101 RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp111 RESNET18/CIFAR100 with lookahead off
if 'run_test' in locals() and run_test == 111:
    la_alpha, linearity_coeff_111 = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp111_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR100')
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=linearity_coeff_111, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=linearity_coeff_111, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay, val_sigma_coeff)

    print(f"Experiment_111 RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp120 RESNET34/CIFAR10 with lookahead on
if 'run_test' in locals() and run_test == 120:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp120_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet34v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay, val_sigma_coeff)

    print(f"Experiment_120 RESNET34/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp121 RESNET34/CIFAR10 with lookahead off
if 'run_test' in locals() and run_test == 121:
    la_alpha, val_linearity_coeff = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp121_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet34v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay, val_sigma_coeff)

    print(f"Experiment_121 RESNET34/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp131 RESNET34/CIFAR100 with lookahead off
if 'run_test' in locals() and run_test == 131:
    la_alpha, linearity_coeff_131 = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp131_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")
    print(f"linearity_coeff={linearity_coeff_131}")

    meta = MetaData(output_dim=100, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR100')
    model = _make_resnet34v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=linearity_coeff_131, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet34v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=linearity_coeff_131, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet34v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_leap = initialise_leap_experiment(clone_resnet34v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)

    print(f"Experiment_131 RESNET34/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_leap, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp140 RESNET18/SVHN
if 'run_test' in locals() and run_test == 140:
    #la_alpha, linearity_coeff_140 = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp140_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('SVHN')
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, \
                                      val_weight_decay, linearity_coeff=val_linearity_coeff, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_sgd = initialise_sgd_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)

    print(f"Experiment_140 RESNET18/SVHN/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp150 RESNET18/ImageNet100
if 'run_test' in locals() and run_test == 150:
    la_alpha, linearity_coeff_150 = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp150_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_ImageNet100()
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=linearity_coeff_150, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=linearity_coeff_150, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_sgd = initialise_sgd_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)

    print(f"Experiment_150 RESNET18/ImageNet100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

#Exp151 RESNET18/TinyImageNet
if 'run_test' in locals() and run_test == 151:
    la_alpha, linearity_coeff_151 = 1.0, 0.1

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp151_{job_nr_str}.log", level=logging.INFO, format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=200, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_ImageNetTiny()
    model = _make_resnet18v2(meta.output_dim).to(device)

    exp_netline = initialise_netline_experiment(model, meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay,\
                                        linearity_coeff=linearity_coeff_151, la_alpha=la_alpha)
    exp_dz = initialise_dz_experiment(clone_resnet18v2(model, meta), meta, EPOCHS_PER_EXPERIMENT, lr1, lr_max, val_momentum, val_weight_decay, \
                                   linearity_coeff=linearity_coeff_151, la_alpha=la_alpha)
    exp_la = initialise_lookahead_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)
    exp_sgd = initialise_sgd_experiment(clone_resnet18v2(model, meta), EPOCHS_PER_EXPERIMENT, lr_max, val_momentum, val_weight_decay)

    print(f"Experiment_151 RESNET18/TinyImageNet/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_la], num_epochs=EPOCHS_PER_EXPERIMENT)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
