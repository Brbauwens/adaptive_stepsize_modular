import torch
from nets.cnn import _make_resnet18v2, _make_resnet34v2
from experiment import ExperimentScheduler as ExperimentWithScheduler
from tools.util_optimisers import initialise_netline, initialise_dz, initialise_lookahead, initialise_leap, initialise_sgd

mask_netline, mask_dz, mask_lookahead, mask_leap, mask_sgd = 1, 2, 4, 8, 16

def initialise_netline_experiment(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                       la_steps = 5, la_alpha = 1.0):
    nl_opt, nl_sch = initialise_netline(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay,\
                                        linearity_coeff=linearity_coeff, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

def initialise_dz_experiment(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                  la_steps = 5, la_alpha = 1.0):
    dz_opt, dz_sch = initialise_dz(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, \
                                   linearity_coeff=linearity_coeff, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("dz", model, dz_opt, dz_sch, do_optimiser_step=False)

def initialise_lookahead_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay, la_steps=5, la_alpha=0.8):
    la_opt, la_sch = initialise_lookahead(model, epochs_per_experiment, lr_max, momentum, weight_decay, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("lookahead", model, la_opt, la_sch, do_optimiser_step=True)

def initialise_leap_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay, sigma_coeff):
    leap_opt, leap_sch = initialise_leap(model, epochs_per_experiment, lr_max, momentum, weight_decay, sigma_coeff)
    return ExperimentWithScheduler("leap", model, leap_opt, leap_sch, do_optimiser_step=True)

def initialise_sgd_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay):
    sgd_opt, sgd_sch = initialise_sgd(model, epochs_per_experiment, lr_max, momentum, weight_decay)
    return ExperimentWithScheduler("sgd-cosine", model, sgd_opt, sgd_sch, do_optimiser_step=True)

def initialise_experiments_by_mask(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay,\
                                   linearity_coeff, la_alpha, sigma_coeff, mask_exp, fn_clone):
    exps = []
    if mask_exp & mask_netline > 0:
        exps.append(initialise_netline_experiment(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay,\
                                            linearity_coeff=linearity_coeff, la_alpha=la_alpha))
    if mask_exp & mask_dz > 0:
        exps.append(initialise_dz_experiment(fn_clone(model, meta), meta, epochs_per_experiment, lr1, lr_max, momentum,\
                                              weight_decay, linearity_coeff=linearity_coeff, la_alpha=la_alpha))
    if mask_exp & mask_lookahead > 0:
        exps.append(initialise_lookahead_experiment(fn_clone(model, meta), epochs_per_experiment, lr_max, momentum, weight_decay))
    if mask_exp & mask_leap > 0:
        exps.append(initialise_leap_experiment(fn_clone(model, meta), epochs_per_experiment, lr_max, momentum,\
                                                weight_decay, sigma_coeff))
    if mask_exp & mask_sgd > 0:
        exps.append(initialise_sgd_experiment(fn_clone(model, meta), epochs_per_experiment, lr_max, momentum, weight_decay))

    return exps

#example of command to run: python -c "from tools.util_experiments import generate_resnet_presaves; generate_resnet_presaves('R18D10')"
BASE_SAVE_DIR = 'saves/nets'
def _resnet_filename(net_type, net_num):
    return BASE_SAVE_DIR+"/"+net_type+"/net_"+str(net_num)+'.pth'

def _make_resnet(net_type):
    if net_type == 'R18D10':
        return _make_resnet18v2(10)
    elif net_type == 'R18D100':
        return _make_resnet18v2(100)
    elif net_type == 'R18D200':
        return _make_resnet18v2(200)
    elif net_type == 'R34D10':
        return _make_resnet34v2(10)
    elif net_type == 'R34D100':
        return _make_resnet34v2(100)
    else:
        raise ValueError(f"Net type not supported: {net_type}")

def generate_resnet_presaves(net_type, presaves_count = 16):
    for cnt in range(presaves_count):
        net = _make_resnet(net_type)
        torch.save(net.state_dict(), _resnet_filename(net_type, cnt))

def load_resnet(net_type, net_num = -1):
    net = _make_resnet(net_type)
    if net_num >= 0:
        net.load_state_dict(torch.load(_resnet_filename(net_type, net_num), weights_only=True))
    net.eval()
    return net
