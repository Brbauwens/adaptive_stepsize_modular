# Storing of quantities by epochs
import math, pickle, numbers, torch
import matplotlib.pylab as plt
from collections import defaultdict
from time import time
import numpy as np

    

class BasicRecorder(defaultdict):
    """A recorder is a dictionary of lists. It stores snapshots in an experiment together with runtimes. 
    The _record function takes a dictionary as input, which represents a snapshot. 
    Values are appended to the items of the same key."""
    def __init__(self):
        super().__init__(list)
        self.restart()

    def restart(self):
        self.tic = time()

    def time(self):
        return time() - self.tic

    def _record(self, quantity_dict):
        for quantity, value in (quantity_dict | {'runtimes' : self.time()}).items():
            self[quantity].append(value)
        self.tic = time()



class Recorder(BasicRecorder):
    """A dictionary that stores information about epochs and minibatches, including runtimes. 
       The main functions are 'record_batch' and 'record_epoch', the latter prints a summary every 'verbose' epochs.
       Minibatch information is stored as a list of lists per epoch.
       """
    def __init__(self, verbose=0):
        super().__init__()
        self.num_epoch, self.verbose = 0, verbose
        self.restart()

    def restart(self):
        self.current_epoch_recorder = BasicRecorder()
        super().restart()

    def record_epoch(self, quantity_dict):
        self.num_epoch += 1
        self._report(quantity_dict)
        super()._record(quantity_dict)
        self['runtimes_batches'].append(self.current_epoch_recorder.pop('runtimes'))
        for quantity, value in self.current_epoch_recorder.items():
            self[quantity].append(value)
        self.restart()

    def record_batch(self, quantity_dict):
        self.current_epoch_recorder._record(quantity_dict)

    def _all_quantities(self):
        return [q for q in self if 'runtime' not in q]

    def _report(self, dct, num_items_shown=9):
        if self.verbose >= 1 and self.num_epoch % self.verbose == 0:
            items = list(dct.items())[:num_items_shown]
            to_str = lambda k,v: f"{v*100:5.2f} %" if 'error' in k else f"{v:7.3}"
            print(f"epoch {self.num_epoch:3d} | time {self.time():6.2f}", end='')
            print(''.join([f' | {k} : {to_str(k,v)}' for k, v in items if type(v) is not list]))
            #with open("experiment_recording.pkl", "wb") as file:   
            #    pickle.dump(self, file)


class ExperimentsRecorder(dict):
    """
    This class is a dictionary of Recordings and is almost useless, except for saving a few lines of code. 

    It is a dictionary of dictionaries that maps: 
    -- methods (for example 'standard SGD' or 'straigth line') and
    -- quantities (for example, 'train accuracy' or '2-norm gradient')
    to a list of real numbers. 
    This list represents the quantity of the method for each epoch.

    Also, for each method, the time is stored when the experiment is finished. 
    """
    def start_experiment(self, experiment_name):
        self.current_experiment = self[experiment_name] = Recorder(verbose=1)
        print(f"\n### Experiment : {experiment_name}")

    def record_epoch(self, quantity_dict):
        self.current_experiment.record_epoch(quantity_dict)

    def record_batch(self, quantity_dict):
        self.current_experiment.record_batch(quantity_dict)

    def time(self):
        return self.current_experiment.time()

    def _all_quantities(self, quantities=[]):
        return list(dict.fromkeys(sum([exp._all_quantities() for exp in self.values()], [])))



def _experiment2color(experiments):
    assert len(experiments) <= 6, "Too many expirements."
    return {experiments : col for experiments, col in zip(experiments, ['b', 'g', 'r', 'c', 'm', 'y'])}


def recplot(experiments_recorder, quantities=[], show_time=False):
    """Main plot function to print all quantities in a Recorder object with number of epoch on the horizontal axis. 
    Also, it can compare recorders from different experiments.
    
    
    Input: experiments_recorder could also be
    - a Trainer object => then its recorder object is used,
    - a Recorder boject => then the dict {'xx' : ...} is used. 
    - an ExperimentsRecorder => then all quantities of all experiments are plotted, 
    and compared when the same ones appear. 

    Minibatch quantities are averaged per epoch. 
    A logscale is used when the ratio of the maximum and the 80% percentile is 4 or more. 
    """
    # Loading across modules gives different Trainer objects, and 'is' gives bugs (at least in ipython). 
    if '.Trainer' in str(type(experiments_recorder)):  
        experiments_recorder = experiments_recorder.recorder
    if '.Recorder' in str(type(experiments_recorder)):
        er = ExperimentsRecorder() 
        er['xx'] = experiments_recorder
        experiments_recorder = er 
    quantities = quantities or experiments_recorder._all_quantities(quantities)
    experiments_recorder = to_numpy(experiments_recorder)  # Transfer from GPU

    fig, axes = plt.subplots(math.ceil(len(quantities)/2), 2, figsize=(18, 4*(int(len(quantities)/2))), constrained_layout=True)
    axes = axes.flatten()
    m2c = _experiment2color(experiments_recorder.keys())
    for i, quantity in enumerate(quantities):
        ax = axes[i]
        ax.set_title(quantity)
        for method, exp in experiments_recorder.items():
            if quantity in exp and (y_vals := exp[quantity]):
                if type(y_vals[0]) is list and y_vals[0]:
                    all_y, y_vals = y_vals, [np.mean(np.array(lst)) for lst in y_vals]
                if type(y_vals) in {list, np.ndarray}:
                    y_vals = np.array(y_vals)  # Is this needed?
                    if np.max(y_vals) > 4*np.percentile(y_vals, 20) :
                        ax.set_yscale('log')
                if show_time:
                    ax.plot(exp.runtimes, y_vals, color=m2c[method], alpha=.5, label=method, marker='o')
                    ax.set_xlim(left=0)
                    ax.set_xlim(right=max([max(r.runtimes) for r in self.values()]))
                else:
                    ax.plot(range(1, len(y_vals)+1), y_vals, color=m2c[method], alpha=1, label=method, marker='o')
                    # If there is only 1 experiment, show detailed plot
                    if len(experiments_recorder) == 1 and "all_y" in locals():
                        for i, y_vals in enumerate(all_y):
                            ax.plot(np.arange(i, i+1, 1.0/len(y_vals)), y_vals, color=m2c[method], alpha=.2, label=method, marker='')
        ax.grid()
        if i == 0 :
            ax.legend()
    if show_time:
        axes[-2].set_xlabel("Time")
        axes[-1].set_xlabel("Time")
    else:
        axes[-2].set_xlabel("Epochs")
        axes[-1].set_xlabel("Epochs")
    #plt.subplots_adjust(hspace=0.3)
    #plt.show(block=False)
    plt.show(block=True)

    # print best scores
    if 'test_error' in quantities and 'test_loss' in quantities:
        for method, exp in experiments_recorder.items():
            vals = min(exp['test_loss']), min(exp['test_error'])
            print(f"{method:<15} : {vals[0]:8.3}   {vals[1]*100:5.2f} %")



def to_numpy(obj):
    if isinstance(obj, numbers.Number):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_numpy(v) for v in obj)
    elif isinstance(obj, set):
        return {to_numpy(v) for v in obj}
    raise Exception("Transfer to cpu and numpy does not work.")




# Test code
"""
from time import sleep
er = ExperimentsRecorder()

er.start_experiment("1st exp")

er.record_batch({'a' : 1, 'b' : 11})
sleep(0.1)
er.record_batch({'a' : 2, 'b' : 12})
sleep(0.1)
er.record_epoch({'q' : 0.1})

er.record_batch({'a' : 1.1, 'b' : 21})
sleep(0.1)
er.record_batch({'a' : 1.2, 'b' : 22})
sleep(0.1)
er.record_epoch({'q' : 0.2})

er.record_batch({'a' : 1.1, 'b' : 31})
sleep(0.1)
er.record_batch({'a' : 1.2, 'b' : 32})
sleep(0.1)
er.record_epoch({'q' : 0.33})


er.start_experiment("2nd exp")

er.record_batch({'a' : 2, 'c' : 15})
sleep(0.1)
er.record_batch({'a' : 3, 'c' : 15})
sleep(0.1)
er.record_epoch({'q' : 0.8})

er.record_batch({'a' : 2, 'c' : 16})
sleep(0.1)
er.record_batch({'a' : 3, 'c' : 16})
sleep(0.1)
er.record_epoch({'q' : 0.8})
"""

# Delete this
#def _gpu2numpy(number):
#    return number.clone().detach().cpu().numpy()
#
