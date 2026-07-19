from tools.recorder import ExperimentsRecorder
from tools.load_data import get_device
from trainer import Trainer

from torch.optim import SGD

device = get_device()

class ExperimentSgd:
    def __init__(self, name, model, quargs_SGD, scheduler_class, quargs_scheduler={}):
        self.name             = name
        self.model            = model
        self.quargs_SGD       = quargs_SGD
        self.optimizer        = SGD(self.model.parameters(), **self.quargs_SGD)
        self.scheduler_class  = scheduler_class 
        self.quargs_scheduler = quargs_scheduler
        self.scheduler = None if scheduler_class is None \
            else scheduler_class(self.optimizer, **self.quargs_scheduler)

def run_experiments(train_dl, test_dl, experiments, num_epochs=2, verbose=False):
    trainers = []
    exp_recorder = ExperimentsRecorder()
    for exp in experiments:
        tn = Trainer(exp.model, exp.optimizer, exp.scheduler)
        trainers.append(tn)
        exp_recorder[exp.name] = tn.recorder

    for tn in trainers:
        tn.recorder.restart()

    for epoch in range(num_epochs):
        if verbose:
            print(f"Epoch {epoch+1} starting")
        for tn in trainers:
            tn.train_loop_init()

        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            for tn in trainers:
                tn.train_step(images, labels)

        if verbose:
            print(f"Epoch {epoch+1} train finished")

        for tn in trainers:
            tn.train_loop_check(test_dl)

        if verbose:
            print(f"Epoch {epoch+1} check finished")

    return exp_recorder, trainers
