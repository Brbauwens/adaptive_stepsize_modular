import torch
from torch.optim import SGD

from tools.recorder import ExperimentsRecorder 
from trainer import Trainer

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Experiment:
    def __init__(self, name, quargs_SGD, scheduler, quargs_scheduler={}):
        self.name             = name
        self.quargs_SGD       = quargs_SGD
        self.scheduler        = scheduler 
        self.quargs_scheduler = quargs_scheduler

def run_experiments(train_dl, test_dl, model, experiments, num_epochs=2):
    exp_recorder  = ExperimentsRecorder()
    initial_state = {k : v.clone() for k,v in model.state_dict().items()}
    trainers = []
    exp_recorder = ExperimentsRecorder()
    for exp in experiments:
        model.load_state_dict(initial_state)
        print('\n' + exp.name)
        optimizer = SGD(model.parameters(), **exp.quargs_SGD)
        scheduler = None if exp.scheduler is None else exp.scheduler(optimizer, **exp.quargs_scheduler)
        t = Trainer(train_dl, test_dl, model, optimizer, scheduler)
        t.train(epochs=num_epochs)
        trainers.append(t)
        exp_recorder[exp.name] = t.recorder
    return exp_recorder, trainers



# =====================
import sys 
from pathlib import Path

if __name__ == '__main__' and Path(sys.argv[0]).stem not in {"ipython", "ipython3", "ipykernel_launcher"}:
    run_test = int(sys.argv[1])


if 'run_test' in locals():
    from torchvision import datasets
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from tools.load_data import load_data
    from nets.basic import BasicTreeLayerNN
    from tools.recorder import recplot
    from lr_schedulers import SPSscheduler, SPScosineScheduler, SPSmaxScheduler



if 'run_test' in locals() and run_test == 2:
    train_dl, test_dl = load_data('PointsDataset')
    model = BasicTreeLayerNN(train_dl, 200, 20).to(device)

    num_epochs = 20
    experiments = [
            Experiment('basic SGD', {'lr' : 0.2}, None, {}), 
            Experiment('SGD + cos', {'lr' : 0.2, 'weight_decay' : 1e-3}, CosineAnnealingLR, {'T_max' : num_epochs}), 
            Experiment('SPS', {'momentum' : 0, 'weight_decay' : 1e-3}, SPSscheduler, {'coeff' : 0.4}), 
            Experiment('SPS + moment', {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPSscheduler, {'coeff' : 0.2}), 
            Experiment('SPSmax + moment', {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPSmaxScheduler, {'coeff' : 0.7}), 
            Experiment('SPScos + moment', {'momentum' : 0.9, 'weight_decay' : 1e-3}, SPScosineScheduler, {'coeff' : 0.2, 'num_epochs' : num_epochs}), 
            #Experiment(
            #    'Net Line 2step', 
            #    {'momentum' : 0}, 
            #    NetLine2StepScheduler, 
            #    {'model' : model, 'alpha' : 0.0078, 'eta_test' : 1e-4, 'beta' : 1e-3}
            #    ), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, model, experiments, num_epochs)
    recplot(rec)


if 'run_test' in locals() and run_test == 3:
    train_dl, test_dl = load_data(datasets.CIFAR10)
    model = BasicTreeLayerNN(train_dl, 200, 20).to(device)

    experiments = [
            Experiment('basic SGD', {'lr' : 0.01}, None, {}), 
            Experiment('SGD +moment +wd', {'lr' : 0.01, 'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 2.5e-4}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            Experiment('SPS coeff=0.04 momentum=0.5', {'momentum' : 0.5}, SPSscheduler, {'coeff' : 0.04}), 
            Experiment('SPS coeff=0.01 moment=0.9', {'momentum' : 0.9}, SPSscheduler, {'coeff' : 0.01}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, model, experiments, num_epochs=5)


if 'run_test' in locals() and run_test == 4:
    from nets.cnn import make_resnet18v2
    train_dl, test_dl = load_data('CIFAR10')
    model = make_resnet18v2(train_dl).to(device)

    experiments = [
            Experiment('basic SGD', {'lr' : 0.01}, None, {}), 
            Experiment('SGD +moment +wd', {'lr' : 0.01, 'momentum' : 0.9, 'nesterov' : True, 'weight_decay' : 2.5e-4}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            Experiment('SPS coeff=0.04 momentum=0.5', {'momentum' : 0.5}, SPSscheduler, {'coeff' : 0.04}), 
            Experiment('SPS coeff=0.01 moment=0.9', {'momentum' : 0.9}, SPSscheduler, {'coeff' : 0.01}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, model, experiments, num_epochs=2)


if 'run_test' in locals() and run_test == 5:
    from nets.cnn import make_resnet34v2
    train_dl, test_dl = load_data('CIFAR100')
    model = make_resnet34v2(train_dl).to(device)

    experiments = [
            Experiment('basic SGD', {'lr' : 0.01}, None, {}), 
            Experiment('SPS coeff=0.05 no_moment', {'momentum' : 0}, SPSscheduler, {'coeff' : 0.08}), 
            ]

    rec, trainers = run_experiments(train_dl, test_dl, model, experiments, num_epochs=1)
