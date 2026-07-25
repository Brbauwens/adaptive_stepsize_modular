import torch

import math

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        self.count += 1
        self.val = val
        self.sum += val

    def avg(self):
        return self.sum / self.count

class AverageCyclicQueue:
    def __init__(self, queue_size, fill_value, device):
        self.queue_size = queue_size
        self.fill_value = fill_value
        self.device = device
        self.reset()

    def reset(self):
        self._pos = 0
        self._pos_cyclic = False
        self._queue = \
            torch.full((self.queue_size,), fill_value=self.fill_value, dtype=torch.float).to(self.device)

    def put_value(self, value):
        self._queue[self._pos] = value
        #pos shift
        self._pos += 1
        if self._pos >= self.queue_size:
            self._pos_cyclic = True
            self._pos = 0

    def get_avg(self):
        return torch.mean(self._queue)

def cosine_annealing2_lr(eta0, eta1, epoch_cos_start, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_finish-epoch_cos_start)))