import torch
from tools.recorder import Recorder
from tools.load_data import get_device

device = get_device()

class Score:
    """ Accumulates loss and error over several chunks. 
    The update() sums the loss over several chunks and returns the loss for the backwards() function.
    """
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self):
        self.mistakes, self.loss, self.count = 0, 0, 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        loss = self.loss_fn(y_pred, y_true)
        self.loss += loss.item()*len(y_true)  # Loss is normalized later. 
        self.mistakes += (y_pred.argmax(dim=1) != y_true).sum().item()
        self.count += len(y_true)
        return loss

    def loss_and_error(self):
        return self.loss/self.count, self.mistakes/self.count

class Trainer:
    """This class implements training in torch with learning rate shedules, in which shedules can be updated in each minibatch.

    The constructor requires:
    -- model, optimizer: a torch model and optimizer
    -- optional, scheduler: any class that has a 'batch_step' or 'step' function. 

    The Recorder class is used to track and plot quantities. 
    The output of the function batch_step of the scheduler controls what is recorded.
    """

    def __init__(self, model, optimizer, scheduler=None, verbose=1, do_optimiser_step=True):
        self.model,    self.optimizer, self.scheduler  =  model.to(device),  optimizer,  scheduler
        self.score_train = None
        self.recorder   =  Recorder(verbose)
        self.do_optimiser_step = do_optimiser_step
        if scheduler:
            assert hasattr(scheduler, 'batch_step') or hasattr(scheduler, 'step'), "Scheduler needs a 'step' or 'batch_step' function."
            assert scheduler.optimizer == optimizer, "scheduler must point to the same optimizer as the Trainer"

    def test(self, test_dl):
        self.model.eval()
        score = Score()
        with torch.no_grad():
            for x, y in test_dl:
                y_pred = self.model(x.to(device))
                score.update(y_pred, y.to(device))
        return score.loss_and_error()

    def _compute_grad(self, x, y, score):
        self.optimizer.zero_grad()
        y_pred = self.model(x).squeeze()
        loss = score.update(y_pred, y)
        loss.backward()
        return loss, y_pred

    def train_loop_init(self):
        self.score_train = Score()
        self.model.train()

    def train_step(self, x, y):
        loss, y_pred = self._compute_grad(x, y, self.score_train)
        quantity_dict = (sch := self.scheduler) is not None and hasattr(sch, 'batch_step') \
                and sch.batch_step(loss=loss, x=x, y=y, y_pred=y_pred) or {} 
        if self.do_optimiser_step:
            self.optimizer.step()

        self.recorder.record_batch(quantity_dict)

    def train_loop_close(self, test_dl):
        if (self.scheduler is not None and hasattr(self.scheduler, 'step')):
            self.scheduler.step()

        self._report(test_dl, self.score_train.loss_and_error())

    def _report(self, test_dl, train_res):
        test_res = self.test(test_dl)
        self.recorder.record_epoch({
               'train_loss' : train_res[0], 'train_error' : train_res[1], 
                'test_loss' : test_res[0],   'test_error' : test_res[1], 
            })


# Code for testing.

from tools.load_data import load_data
from torch.optim import SGD
from optim.lr_schedulers import SPSmaxScheduler

if 'run_test' in locals() and run_test >= 1 :
    from nets.basic import BasicThreeLayerNN

    train_dl, test_dl = load_data('PointsDataset')
    model = BasicThreeLayerNN(train_dl, 200, 20).to(device)
    optim = SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-3)

    sched = SPSmaxScheduler(optim, coeff=0.5)
    t = Trainer(train_dl, test_dl, model, optim, sched)
    t.train(20)


"""
if 'compute' in locals() and compute:
    #train_dl, test_dl = load_data('FashionMNIST')
    if "_data_loaded" not in globals() or _data_loaded != 'FashionMNIST':
        train_dl, test_dl = load_data('FashionMNIST')
        _data_loaded = 'FashionMNIST'

    #if "_data_loaded" not in globals() or _data_loaded != 'CIFAR10':
    #    (train_dl, test_dl), _data_loaded = load_data('CIFAR10'), 'CIFAR10'
    from nets.basic import BasicTreeLayerNN #, make_resnet18v2
    model = BasicTreeLayerNN(train_dl, 300, 50)
    #model = make_resnet18v2(train_dl)
    from torch.optim import SGD
    optim = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-3)
    #from lr_schedulers import SPSmaxScheduler #NetLine2StepScheduler, StraightLineScheduler
    #from torch.optim.lr_scheduler import CosineAnnealingLR
    #scheduler = SPSmaxScheduler(optim)
    #schedulerNetLine = NetLine2StepScheduler(optim, model, alpha=0.6, beta=1e-4, eta_test=5e-5)
    num_epochs = 10
    from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
    scheduler = CosineAnnealingLR(optim, T_max=num_epochs)
    from lr_schedulers import SPSscheduler
    scheduler = SPSscheduler(optim, coeff=0.1, eps=0.01)
    #from try_many_lrs import show_lrs, BruteForceScheduler
    #scheduler = BruteForceScheduler(optim)
    #t = Trainer(train_dl, test_dl, model, optim, scheduler, method_name='x', report_fn_step=show_lrs)
    print('new trainer t')
    t = Trainer(train_dl, test_dl, model, optim, scheduler)
    #t.train(epochs=num_epochs)
"""


"""
    from torchvision import datasets, transforms, models
    #from load_data import load_data
    from torch.utils.data import DataLoader
    from nets import BasicNN, simple_cnn, make_resnet9
    from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
    from torch.optim import SGD
    from lr_schedulers import SPSmaxScheduler #NetLine2StepScheduler, StraightLineScheduler
    import torch.nn as nn
    #from storing_results import ResultsStorageByEpochs


    ## Load a pre-trained ResNet18 model
    #model = models.resnet18(pretrained=True)
    ## Modify the first convolutional layer to accept 1 input channel
    #model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    ## Get the number of input features to the original final layer
    #num_ftrs = model.fc.in_features
    ## Replace the final fully connected layer with a new one for 10 classes
    #model.fc = nn.Linear(num_ftrs, 10)

    #from torchvision.models import resnet18
    #model = resnet18(3, 10).to(device)
        if verbose >= 1 and num_epoch % verbose == 0:
            s = [f"{loss:.2f} {err*100:.2f} %" for loss, err in [train_res, test_res]]
            print(f"epoch {num_epoch:3d} | time {self.recorder.time():.2f} | train {s[0]} | test {s[1]}")
    """


"""
    def _report_step_fn(self, loss):
        instructions = {
                'lr' : self.optimizer.param_groups[0]['lr'], 
                'grad_norm' : square_gradnorm(self.optimizer)
                'batch_loss' : loss
                }
        return {key : val for key, val in instructions.items() if key in self._report_step_quantities}
"""

