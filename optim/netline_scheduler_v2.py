import torch
from torch import Tensor
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging
import numpy as np
from typing import List, Optional

from optim.utils import AverageMeter, AverageCyclicQueue, cosine_annealing2_lr

#force_trainmode=True/False
def snl_forward(net, images, force_evalmode):
    if force_evalmode == True:
        training = net.training
        net.train(False)
        logging.info("##Snl: --==Explicit train forward==--")
        logits = net.forward(images)
        net.train(training)
        return logits
    else:
        return net.forward(images)

def eta(eta_test, delta_pq, delta_qq, norm_pq, norm_qq, epsilon, beta_min, do_logging):
    cos_phi = torch.sum(delta_pq*delta_qq)/torch.maximum(norm_pq*norm_qq, epsilon)
    eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
    if do_logging:
        logging.info("##Snl: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                    .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
    return eta_next, cos_phi

class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta, foreach=False):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta
        self.foreach = foreach
        #self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction='mean')
        self._eye = torch.eye(meta.output_dim, dtype=torch.float).to(meta.device)
        self._zero = torch.tensor(0.0).to(meta.device)
        self._one = torch.tensor(1.0).to(meta.device)

        self.beta_min = torch.tensor(1e-10).to(meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = torch.tensor(1e-20).to(meta.device)
        self.y_part = 0.0

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_shorten_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.alpha_momentum = 1.0

        self.lr_averaging_check_up = 1.0
        self.lr_averaging_check_down = 1.0
        self.lr_averaging_queue_size = 100
        self._lr_averaging_queue = None

        self.alpha_nomomentum = 0.75
        self.alpha_nomomentum_max = self._one
        self.alpha_nomomentum_queue_size = 100
        self._alpha_nomomentum_queue = None

        self._flag_check_no_backstep = False

        self.lr0 = 1e-5
        self.lr_max = 2e-2
        self.lr_sample = 2e-2

        self.epochs_per_experiment = 50
        self.epochs_warmup = 3
        self.epochs_sampling = -1
        self.epochs_wide = -1
        self.epochs_middle = -1
        self.epochs_alpha_nomomentum_average = 5

        self._epoch = 0
        self._sample_prob = 0.05

        self._epochs_alpha_nomomentum_average_arr_pos = 0
        self._epoch_alpha_nomomentum_meter = AverageMeter()

    def init_params(self):
        momentum = self.optimizer.param_groups[0]['momentum']
        self.alpha_momentum = math.sqrt(1-momentum**2) \
            if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        self.step(epoch = -1)
        self._epochs_alpha_nomomentum_average_arr = np.zeros(self.epochs_alpha_nomomentum_average)
        self._epochs_alpha_nomomentum_average_arr_pos = 0

    def put_epochs_alpha_nomomentum_average(self, value):
        self._epochs_alpha_nomomentum_average_arr[self._epochs_alpha_nomomentum_average_arr_pos] = value
        self._epochs_alpha_nomomentum_average_arr_pos += 1
        if self._epochs_alpha_nomomentum_average_arr_pos >= self.epochs_alpha_nomomentum_average:
            self._epochs_alpha_nomomentum_average_arr_pos = 0

    def init_eta_averaging(self):
        self._lr_averaging_queue = AverageCyclicQueue(queue_size = self.lr_averaging_queue_size, fill_value = 0.02, device = self.meta.device)

    def calc_eta_averaging(self, eta):
        self._lr_averaging_queue.put_value(eta)

        if (self.lr_averaging_check_up >= 1.0 and self.lr_averaging_check_down >= 1.0):
            return eta

        if not self._lr_averaging_queue._pos_cyclic:
            return eta
        else:
            eta_avg = self._lr_averaging_queue.get_avg()
            if (self.lr_averaging_check_up <= 0.0 and self.lr_averaging_check_down <= 0.0):
                return eta_avg
            else:
                eta_delta = eta - eta_avg
                return eta_avg + eta_delta*torch.where(torch.sign(eta_delta) == self._one, self.lr_averaging_check_up, self.lr_averaging_check_down)

    def init_alpha_nomomentum_averaging(self):
        self._alpha_nomomentum_queue = AverageCyclicQueue(queue_size = self.alpha_nomomentum_queue_size, fill_value = 0.75, device = self.meta.device)

    def calc_alpha_nomomentum_averaging(self, alpha_nomomentum):
        self._alpha_nomomentum_queue.put_value(alpha_nomomentum)

        if not self._alpha_nomomentum_queue_pos_cyclic:
            return alpha_nomomentum
        else:
            return torch.mean(self._alpha_nomomentum_queue)

    def step_nl_with_loss_fn(self, labels, images, loss_fn, is_sample_step):
        """Method to call in every minibatch with external scheduler/optimiser params adjustment in minibatch-cycle (no step call)
        """
        net = self.net
        optimizer = self.optimizer

        logging.info("##Snl: Step start calculating logits and qq0")
        net.zero_grad()
        logitsG = snl_forward(net, images, force_evalmode=False) ## new gradient with dropout is generated here (1*)
        logging.info("##Snl: calculating criterion")
        loss = loss_fn.forward(logitsG, labels)
        logging.info("##Snl: performing small step")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            return self._step_nl(labels, images, logitsG, is_sample_step)

    def step(self, epoch = None):
        """Method to call in every epoch for scheduler/optimiser params adjustment. It is called after the epoch's training loop
        """
        if epoch is not None:
            self._epoch = epoch

        self.put_epochs_alpha_nomomentum_average(self._epoch_alpha_nomomentum_meter.avg())

        self._epoch_alpha_nomomentum_meter.reset()
        self._epoch += 1

        if self._epoch < self.epochs_wide:
            self.lr_averaging_check_down = 0.5
            self.lr_averaging_check_up = 0.5
        elif epoch < self.epochs_middle:
            self.lr_averaging_check_down = 0.25
            self.lr_averaging_check_up = 0.25
        else:
            self.lr_averaging_check_down = 0.05
            self.lr_averaging_check_up = 0.05

        self.lr_sample = cosine_annealing2_lr(self.lr_max, 0.0, 0, self.epochs_per_experiment, self._epoch)
        if self._epoch <= self.epochs_sampling:
            self._sample_prob = 0.05
        else:
            self._sample_prob = 0.0
            alpha_nomomentum_nocosine_base = np.average(self._epochs_alpha_nomomentum_average_arr)
            self.alpha_nomomentum = \
                cosine_annealing2_lr(alpha_nomomentum_nocosine_base, 0.0, self.epochs_sampling, self.epochs_per_experiment, self._epoch)

    def batch_step(self, x, y, y_pred):
        """Method to call in every minibatch together with step call in every epoch. Loss forward-backward performed externally
        """

        images, labels, logitsG = x, y, y_pred
        optimizer = self.optimizer

        is_sample_step = self._epoch < self.epochs_warmup or self._alpha_nomomentum_queue._pos_cyclic == False or np.random.binomial(n=1, p=self._sample_prob) == 1
        if is_sample_step:
            optimizer.param_groups[0]['lr'] = self.lr_sample
        else:
            optimizer.param_groups[0]['lr'] = self.lr0

        optimizer.step()

        with torch.no_grad():
            result = self._step_nl(labels, images, logitsG, is_sample_step)
            self._epoch_alpha_nomomentum_meter.update(result['alpha_nomomentum'])
            return result

    def _step_nl(self, labels, images, logitsG, is_sample_step):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer
        eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size

        logits0 = (logitsG if self.dropout_mode == False else snl_forward(net, images, force_evalmode=self.dropout_mode))

        logging.info("##Snl: , calculating pp")
        pp = F.one_hot(labels, meta.output_dim)
        qq0 = F.softmax(logits0, dim=1) ## all qqxx calculated with dropout-off/eval-mode
        logging.info("##Snl: calculating learning rate")
        logits1 = snl_forward(net, images, force_evalmode=self.dropout_mode)
        qq1 = F.softmax(logits1, dim=1) #0-point, 1-neuron?
        delta_pq, delta_qq1 = pp-qq0, qq1-qq0

        logging.info("##Snl: calculating eta_preactivation")
        eta2_raw_y = 0.0
        if self.y_part > 0.0:
            dz = (logits1-logits0)/eta1
            qqq = qq0[:,:,None]*(self._eye[None,:,:]-qq0[:,None,:])
            eta2_raw_y = torch.squeeze(torch.sum(delta_pq*dz)/torch.sum(dz[:,:,None]*qqq*dz[:,None,:]))

        logging.info("##Snl: calculating eta_analytic_n2")
        norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
        eta2_raw, cos_phi = eta(eta1, delta_pq, delta_qq1, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.do_logging)

        eta2_momentum = ((1.0 - self.y_part)*eta2_raw + self.y_part * eta2_raw_y)*self.alpha_momentum
        if (is_sample_step):
            eta2 = eta2_pre = optimizer.param_groups[0]['lr']
            self.alpha_nomomentum = torch.minimum(self.calc_alpha_nomomentum_averaging(eta2_pre/eta2_momentum), self.alpha_nomomentum_max)
        else:
            eta2_pre = eta2_momentum*self.alpha_nomomentum
            eta2 = self.calc_eta_averaging(eta2_pre)

        if self.do_logging:
            logging.info("##Snl: alpha_epoch={}, alpha_momentum={}, eta2_pre={}, eta2={}".format(self.alpha_epoch, self.alpha_momentum, eta2_pre, eta2))
        logging.info("##Snl: shifting params to the rest of step")

        for group in optimizer.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = optimizer._init_group(
                group, params, grads, momentum_buffer_list
            )
            if self.foreach == False:
                for num, param in enumerate(params):
                    grad, momentum_buffer = grads[num], momentum_buffer_list[num]

                    if not is_sample_step:
                        if (not self._flag_check_no_backstep or eta2 > self._zero):
                            eta2_shift = eta2.add(-eta1)
                            buffer_x_shift = None
                            if group["momentum"] == 0:
                                buffer_x_shift = grad.mul(-eta2_shift)
                            else:
                                buffer_x_shift = momentum_buffer.mul(-eta2_shift)
                            param.add_(buffer_x_shift)

            else:
                if not is_sample_step:
                    if (not self._flag_check_no_backstep or eta2 > self._zero):
                        eta2_shift = eta2.add(-eta1)
                        buffers_x_shift = None
                        if group["momentum"] == 0:
                            buffers_x_shift = torch._foreach_mul(grads, -eta2_shift)
                        else:
                            buffers_x_shift = torch._foreach_mul(momentum_buffer_list, -eta2_shift)
                        torch._foreach_add_(params, buffers_x_shift)

        logging.info("####Snl: step finish, returning step_result")
        return self.step_results(eta2, eta2_pre, norm_pq, norm_qq1, cos_phi, self.alpha_nomomentum*self.alpha_momentum, self.alpha_nomomentum, qq0)

    def step_results(self, eta2, eta2_pre, norm_pq, norm_qq1, cos_phi, alpha_full, alpha_nomomentum, qq0):
        result = {}
        result['eta2'] = eta2
        result['eta2_pre'] = eta2_pre
        result['norm_pq'] = norm_pq
        result['norm_qq1'] = norm_qq1
        result['cos_phi'] = cos_phi
        result['alpha_full'] = alpha_full
        result['alpha_nomomentum'] = alpha_nomomentum
        result['qq0'] = qq0

        return result
