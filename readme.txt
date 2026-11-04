Experiment pre-sets are placed in experiments_pack2.py. There are the following experiments:

100 RESNET18/CIFAR10 with lookahead on
101 RESNET18/CIFAR10 with lookahead off
110 RESNET18/CIFAR100 with lookahead on
111 RESNET18/CIFAR100 with lookahead off
120 RESNET34/CIFAR10 with lookahead on
121 RESNET34/CIFAR10 with lookahead off
130 RESNET34/CIFAR100 with lookahead on
131 RESNET34/CIFAR100 with lookahead off
140 RESNET18/SVHN with lookahead on
150 RESNET18/ImageNet100 with lookahead off
151 RESNET18/TinyImageNet with lookahead off

Every experiment include 5 optimisers to start in parallel: netline, dz, lookahead, leap, sgd. To run an experiment, the following command must be used:

python experiments_pack2.py <experiment_num> <job_number> <epochs_number> <optimisers_mask> <manual_seed>
experiment_num: from the list above
job_number: some unique string id to use in log-file name as a suffix
epochs_number: number of epochs in the run
optimisers_mask: bitmask for optimisers. The optimisers have the following ids: netline = 1, dz = 2, lookahead = 4, leap = 8, sgd = 16.
                 The bitmask by bitwise and with the ids excludes the optimisers. For example, optimisers_mask = 15 will exclude sgd. 
                 If the bitmask not set, all optimisers are included.
manual_seed: value from 0 to 15 to use fixed net initialisation and data loader. If not set, seed is not fixed.

Example of command to run experiment 100:

python experiments_pack2.py 100 146 50 15 0
