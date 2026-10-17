from torch import optim

import os
import logging
from datetime import datetime

from tools.load_data import get_device, load_data
from experiment import ExperimentScheduler as ExperimentWithScheduler, run_experiments
from optim.optimiser_netline_v3 import NetLine
from optim.optimiser_dz_v3 import NetDz
from optim.scheduler_netline import CosineAnnealingNetLine
from optim.utils import MetaData
from nets.cnn import _make_resnet18v2, _make_resnet34v2
from optim.optimizer_pack1 import Lookahead

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
val_momentum = 0.9
la_steps=5
la_alpha=0.8

def format_params():
    return f"lr1={lr1}, lr_max={lr_max}, momentum={val_momentum}, la_steps={la_steps}, la_alpha={la_alpha}"

if 'run_test' in locals() and run_test == 100:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp100_{job_nr_str}.log",
                    level=logging.INFO,
                    format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet18v2(meta.output_dim).to(device)

    #Netline
    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_netline = ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

    #Dz
    model_dz = clone_resnet18v2(model, meta)

    dz_opt = NetDz(model=model_dz, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_dz = ExperimentWithScheduler("dz", model_dz, dz_opt, dz_sch, do_optimiser_step=False)

    #Lookahead
    model_lookahead = clone_resnet18v2(model, meta)
    opt_sgd_lookahead = optim.SGD(model_lookahead.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    opt_lookahead = Lookahead(opt_sgd_lookahead)
    schd_lookahead = optim.lr_scheduler.CosineAnnealingLR(opt_lookahead, T_max=EPOCHS_PER_EXPERIMENT)
    exp_lookahead = ExperimentWithScheduler("lookahead", model_lookahead, opt_lookahead, schd_lookahead, do_optimiser_step=True)

    #Sgd
    model_sgd = clone_resnet18v2(model, meta)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)
    exp_sgd = ExperimentWithScheduler("sgd-cosine", model_sgd, opt_sgd, schd_sgd, do_optimiser_step=True)

    print(f"Experiment RESNET18/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with params:\r\n" + format_params())
    rec, trainers = run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_lookahead],\
                                     num_epochs=EPOCHS_PER_EXPERIMENT, verbose=True)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

if 'run_test' in locals() and run_test == 110:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp110_{job_nr_str}.log",
                    level=logging.INFO,
                    format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR100')
    model = _make_resnet18v2(meta.output_dim).to(device)

    #Netline
    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_netline = ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

    #Dz
    model_dz = clone_resnet18v2(model, meta)

    dz_opt = NetDz(model=model_dz, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_dz = ExperimentWithScheduler("dz", model_dz, dz_opt, dz_sch, do_optimiser_step=False)

    #Lookahead
    model_lookahead = clone_resnet18v2(model, meta)
    opt_sgd_lookahead = optim.SGD(model_lookahead.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    opt_lookahead = Lookahead(opt_sgd_lookahead)
    schd_lookahead = optim.lr_scheduler.CosineAnnealingLR(opt_lookahead, T_max=EPOCHS_PER_EXPERIMENT)
    exp_lookahead = ExperimentWithScheduler("lookahead", model_lookahead, opt_lookahead, schd_lookahead, do_optimiser_step=True)

    #Sgd
    model_sgd = clone_resnet18v2(model, meta)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)
    exp_sgd = ExperimentWithScheduler("sgd-cosine", model_sgd, opt_sgd, schd_sgd, do_optimiser_step=True)

    print(f"Experiment RESNET18/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with params:\r\n" + format_params())
    rec, trainers = run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_lookahead],\
                                     num_epochs=EPOCHS_PER_EXPERIMENT, verbose=True)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

if 'run_test' in locals() and run_test == 120:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp120_{job_nr_str}.log",
                    level=logging.INFO,
                    format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR10')
    model = _make_resnet34v2(meta.output_dim).to(device)

    #Netline
    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_netline = ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

    #Dz
    model_dz = clone_resnet34v2(model, meta)

    dz_opt = NetDz(model=model_dz, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_dz = ExperimentWithScheduler("dz", model_dz, dz_opt, dz_sch, do_optimiser_step=False)

    #Lookahead
    model_lookahead = clone_resnet34v2(model, meta)
    opt_sgd_lookahead = optim.SGD(model_lookahead.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    opt_lookahead = Lookahead(opt_sgd_lookahead)
    schd_lookahead = optim.lr_scheduler.CosineAnnealingLR(opt_lookahead, T_max=EPOCHS_PER_EXPERIMENT)
    exp_lookahead = ExperimentWithScheduler("lookahead", model_lookahead, opt_lookahead, schd_lookahead, do_optimiser_step=True)

    #Sgd
    model_sgd = clone_resnet34v2(model, meta)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)
    exp_sgd = ExperimentWithScheduler("sgd-cosine", model_sgd, opt_sgd, schd_sgd, do_optimiser_step=True)

    print(f"Experiment RESNET34/CIFAR10/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with params:\r\n" + format_params())
    rec, trainers = run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_lookahead],\
                                     num_epochs=EPOCHS_PER_EXPERIMENT, verbose=True)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

if 'run_test' in locals() and run_test == 130:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp130_{job_nr_str}.log",
                    level=logging.INFO,
                    format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=100, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('CIFAR100')
    model = _make_resnet34v2(meta.output_dim).to(device)

    #Netline
    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_netline = ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

    #Dz
    model_dz = clone_resnet34v2(model, meta)
    dz_opt = NetDz(model=model_dz, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_dz = ExperimentWithScheduler("dz", model_dz, dz_opt, dz_sch, do_optimiser_step=False)

    #Lookahead
    model_lookahead = clone_resnet34v2(model, meta)
    opt_sgd_lookahead = optim.SGD(model_lookahead.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    opt_lookahead = Lookahead(opt_sgd_lookahead)
    schd_lookahead = optim.lr_scheduler.CosineAnnealingLR(opt_lookahead, T_max=EPOCHS_PER_EXPERIMENT)
    exp_lookahead = ExperimentWithScheduler("lookahead", model_lookahead, opt_lookahead, schd_lookahead, do_optimiser_step=True)

    #Sgd
    model_sgd = clone_resnet34v2(model, meta)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)
    exp_sgd = ExperimentWithScheduler("sgd-cosine", model_sgd, opt_sgd, schd_sgd, do_optimiser_step=True)

    print(f"Experiment RESNET34/CIFAR100/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with params:\r\n" + format_params())
    rec, trainers = run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_lookahead],\
                                     num_epochs=EPOCHS_PER_EXPERIMENT, verbose=True)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

if 'run_test' in locals() and run_test == 140:

    job_nr_str = get_job_nr(sys.argv)
    logging.basicConfig(filename=f"logs/exp140_{job_nr_str}.log",
                    level=logging.INFO,
                    format="%(levelname)s: %(asctime)s %(message)s")

    meta = MetaData(output_dim=10, device=device)
    EPOCHS_PER_EXPERIMENT = get_epochs_per_exp(sys.argv)

    train_dl, test_dl = load_data('SVHN')
    model = _make_resnet18v2(meta.output_dim).to(device)

    #Netline
    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_netline = ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

    #Dz
    model_dz = clone_resnet18v2(model, meta)
    dz_opt = NetDz(model=model_dz, meta=meta, lr1=lr1, momentum=val_momentum, weight_decay=5e-3, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = EPOCHS_PER_EXPERIMENT, epochs_warmup = 0, epochs_shutdown = 0)
    exp_dz = ExperimentWithScheduler("dz", model_dz, dz_opt, dz_sch, do_optimiser_step=False)

    #Lookahead
    model_lookahead = clone_resnet18v2(model, meta)
    opt_sgd_lookahead = optim.SGD(model_lookahead.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    opt_lookahead = Lookahead(opt_sgd_lookahead)
    schd_lookahead = optim.lr_scheduler.CosineAnnealingLR(opt_lookahead, T_max=EPOCHS_PER_EXPERIMENT)
    exp_lookahead = ExperimentWithScheduler("lookahead", model_lookahead, opt_lookahead, schd_lookahead, do_optimiser_step=True)

    #Sgd
    model_sgd = clone_resnet18v2(model, meta)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=lr_max, momentum=val_momentum, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)
    exp_sgd = ExperimentWithScheduler("sgd-cosine", model_sgd, opt_sgd, schd_sgd, do_optimiser_step=True)

    print(f"Experiment RESNET18/SVHN/{EPOCHS_PER_EXPERIMENT} epochs №{job_nr_str} "+
          f"started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with params:\r\n" + format_params())
    rec, trainers = run_experiments(train_dl, test_dl, [exp_netline, exp_dz, exp_sgd, exp_lookahead],\
                                     num_epochs=EPOCHS_PER_EXPERIMENT, verbose=True)
    print(f"Experiment {job_nr_str} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
