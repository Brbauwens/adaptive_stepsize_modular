
import torch, math
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
assert device == 'cuda'


@torch.no_grad()
def square_gradnorm(optimizer) -> torch.Tensor:
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    return clip_grad_norm_(params, max_norm=1e9)


# ======== SPS - Stochastic Polyak Stepsize ============

class AdaptiveScheduler(dict):
    """Every adaptive scheduler must have a 'compute_lr' function.  
    The scheduler *is* a dictionary and each item is recorderded by the trainer class for easy plotting."""
    def batch_step(self, loss, *args, **kwargs):
        self.optimizer.param_groups[0]['lr'] = self['batch_lr'] = self.compute_lr(loss, *args, **kwargs)
        return self


class SPSscheduler(AdaptiveScheduler):
    """Learning rate obtained from linear approximation of the loss function. 
    In practice, this leads to an over estimation of good rates, and this is metigated with 'coeff'."""
    def __init__(self, optimizer, coeff=0.2, eps=1e-3):
        self.optimizer = optimizer
        self.coeff, self.eps = coeff, eps
        self.times_small_norm = 0

    def compute_lr(self, loss, *args, **kwargs):
        self['loss'] = loss
        self['square_gradnorm'] = sq_norm = square_gradnorm(self.optimizer)
        self.times_small_norm += 1 if sq_norm < self.eps else 0
        return self.coeff * loss / max(self.eps, sq_norm)


class SPSmaxScheduler(SPSscheduler):
    def __init__(self, optimizer, coeff=0.2, eps=1e-3):
        super().__init__(optimizer, coeff, eps)
        self.multiplier = 1.2    #2**(1/len(train_dl)) was suggested in the paper, but is way to small
        self['batch_lr'] = 1e16   #Initialize the 'previous' learning rate

    def compute_lr(self, loss, *args, **quargs):
        lrs = self['batch_lr'] * self.multiplier, self.coeff * super().compute_lr(loss) 
        self['is_new_lr'] = int(lrs[0] >= lrs[1])
        return min(lrs)


class SPScosineScheduler(SPSscheduler):
    def __init__(self, optimizer, num_epochs, coeff=0.2, eps=1e-3):
        super().__init__(optimizer, coeff, eps)
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.initial_coeff = self.coeff = coeff

    def step(self):
        self.coeff = 0.5 * self.initial_coeff * math.cos(math.pi * self.current_epoch / self.num_epochs)
