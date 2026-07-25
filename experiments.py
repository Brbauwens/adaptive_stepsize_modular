import torch
import os

from tools.load_data import get_device
from experiment import ExperimentSgd as Experiment, run_experiments
from nets.cnn import make_resnet18v2, make_resnet34v2

device = get_device()

# =====================
import sys 
from pathlib import Path

if __name__ == '__main__' and Path(sys.argv[0]).stem not in {"ipython", "ipython3", "ipykernel_launcher"}:
    run_test = int(sys.argv[1])

if 'run_test' in locals():
    from torchvision import datasets
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from tools.load_data import load_data
    from nets.basic import BasicThreeLayerNN
    from tools.recorder import recplot
    from optim.lr_schedulers import SPSscheduler, SPScosineScheduler, SPSmaxScheduler

    os.makedirs(".data_experiments", exist_ok=True)


if 'run_test' in locals() and run_test == 2:
    exp2_save_file = '.data_experiments/exp2.pth'
    def clone_model2():
        mdl = BasicThreeLayerNN(train_dl, 200, 20).to(device)
        mdl.load_state_dict(torch.load(exp2_save_file, weights_only=True))
        return mdl

    train_dl, test_dl = load_data('PointsDataset')
    model = BasicThreeLayerNN(train_dl, 200, 20).to(device)
    torch.save(model.state_dict(), exp2_save_file)

    num_epochs = 2
    experiments = [
            Experiment('basic SGD', model, {'lr' : 0.2}, None, {}), 
            Experiment('SGD + cos', clone_model2(), {'lr' : 0.2, 'weight_decay' : 1e-3}, CosineAnnealingLR, {'T_max' : num_epochs}), 
            Experiment('SPS', clone_model2(), {'momentum' : 0, 'weight_decay' : 1e-3}, SPSscheduler, {'coeff' : 0.4}), 
            Experiment('SPS + moment', clone_model2(), {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPSscheduler, {'coeff' : 0.2}), 
            Experiment('SPSmax + moment', clone_model2(), {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPSmaxScheduler, {'coeff' : 0.7}), 
            Experiment('SPScos + moment', clone_model2(), {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPScosineScheduler, {'coeff' : 0.2, 'num_epochs' : num_epochs}), 
            #Experiment(
            #    'Net Line 2step', 
            #    {'momentum' : 0}, 
            #    NetLine2StepScheduler, 
            #    {'model' : model, 'alpha' : 0.0078, 'eta_test' : 1e-4, 'beta' : 1e-3}
            #    ), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, experiments, num_epochs)
    recplot(rec)


if 'run_test' in locals() and run_test == 3:
    exp3_save_file = '.data_experiments/exp3.pth'
    def clone_model3():
        mdl = BasicThreeLayerNN(train_dl, 200, 20).to(device)
        mdl.load_state_dict(torch.load(exp3_save_file, weights_only=True))
        return mdl

    train_dl, test_dl = load_data(datasets.CIFAR10)
    model = BasicThreeLayerNN(train_dl, 200, 20).to(device)
    torch.save(model.state_dict(), exp3_save_file)

    experiments = [
            Experiment('basic SGD', model, {'lr' : 0.01}, None, {}), 
            Experiment('SGD +moment +wd', clone_model3(), {'lr' : 0.01, 'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 2.5e-4}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', clone_model3(), {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            Experiment('SPS coeff=0.04 momentum=0.5', clone_model3(), {'momentum' : 0.5}, SPSscheduler, {'coeff' : 0.04}), 
            Experiment('SPS coeff=0.01 moment=0.9', clone_model3(), {'momentum' : 0.9}, SPSscheduler, {'coeff' : 0.01}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, experiments, num_epochs=5, verbose=True)

if 'run_test' in locals() and run_test == 4:
    exp4_save_file = '.data_experiments/exp4.pth'
    def clone_model4():
        mdl = make_resnet18v2(train_dl).to(device)
        mdl.load_state_dict(torch.load(exp4_save_file, weights_only=True))
        return mdl

    train_dl, test_dl = load_data('CIFAR10')
    model = make_resnet18v2(train_dl).to(device)
    torch.save(model.state_dict(), exp4_save_file)

    experiments = [
            Experiment('basic SGD', model, {'lr' : 0.01}, None, {}), 
            Experiment('SGD +moment +wd', clone_model4(), {'lr' : 0.01, 'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 2.5e-4}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', clone_model4(), {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            Experiment('SPS coeff=0.04 momentum=0.5', clone_model4(), {'momentum' : 0.5}, SPSscheduler, {'coeff' : 0.04}), 
            Experiment('SPS coeff=0.01 moment=0.9', clone_model4(), {'momentum' : 0.9}, SPSscheduler, {'coeff' : 0.01}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, experiments, num_epochs=2, verbose=True)


if 'run_test' in locals() and run_test == 5:
    exp5_save_file = '.data_experiments/exp5.pth'
    def clone_model5():
        mdl = make_resnet18v2(train_dl).to(device)
        mdl.load_state_dict(torch.load(exp5_save_file, weights_only=True))
        return mdl

    train_dl, test_dl = load_data('CIFAR100')
    model = make_resnet18v2(train_dl).to(device)
    torch.save(model.state_dict(), exp5_save_file)

    experiments = [
            Experiment('basic SGD', model, {'lr' : 0.01}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', clone_model5(), {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, experiments, num_epochs=1, verbose=True)
