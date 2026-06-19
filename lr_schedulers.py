
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




# ======== Netline 2step scheduler ============

# !!!!!!!!!!! it is not correct (and old)

def dot(u,v):
    assert u.shape == v.shape
    return (u*v).sum().item()

class NetLine2StepScheduler:
    """Learning rate obtained from linear approximation of the loss function. See the paper XXX."""
    def __init__(self, optimizer, model, eta_test=1e-4, alpha=0.33, beta=1e-6, alpha_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.eta_test = eta_test
        self.alpha = alpha
        self.beta = beta
        self.alpha_scheduler = alpha_scheduler
        self.softmax = torch.nn.Softmax(dim=-1)
        self.info = []
        self.no_step_calls = 0

    def step(self):
        '''Executed every epoch.'''
        if self.alpha_scheduler:
            self.alpha_scheduler.step()

    def _step_optim(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.step()

    def _get_test_softmax(self, x, eta_test):
        self._step_optim(eta_test)
        q_test = self.softmax(self.model(x)) 
        self._step_optim(-eta_test) # I suspect that updating weights is faster than copying them.
        return q_test

    def _calculate_eta(self, x, y, y_pred):
        with torch.no_grad():
            q, y_1hot = self.softmax(y_pred), one_hot(y, y_pred.shape[1])
            diff_q = self._get_test_softmax(x, self.eta_test) - q
            # But while debugging I had norm(diff_q), so I temporarily changed it.
            eta = self.eta_test * self.alpha * dot(y_1hot - q, diff_q) / (norm(diff_q) + self.beta)**2
            if eta < 1e-4 :
                print('.', end='')
                return 0.01
                #print("probaly something is wrong, brute forcing etas for debugging in variable 'bf' and 'etas'")
                #import matplotlib.pylab as mpl
                #etas = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1]
                #bf = self._brute_force_etas(x, q, etas)
                #1/0
            #For monitoring purposes, the cosine between diff_q and diff_q_y below. 
            diff_q_y = y_1hot - q
            self.info.append(( eta.item(), (dot(diff_q, diff_q_y) / norm(diff_q) / norm(diff_q_y)).item(), norm(diff_q).item() ))
            return eta.item()

    def _brute_force_etas(self, x, q, etas):
        return [norm(self._get_test_softmax(x, eta) - q).item() for eta in etas]

    def batch_step(self, loss, x, y, y_pred):
        self.no_step_calls += 1
        self.optimizer.param_groups[0]['lr'] = self._calculate_eta(x, y, y_pred)
        # print(f"eta = {eta:.4f}")
        # Recall that optimizer.step() is done in the mainloop of Trainer

