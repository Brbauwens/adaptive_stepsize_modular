# Stepsize modulation for cross entropy

This repository contains implementations for [Stepsize modulation for cross entropy from local quadratic optimization](https://arxiv.org/abs/2611.05191) in PyTorch.

Two algoritms so-dz and cs-dq are presented. Algorithms decrease the loss faster than the schedule that is perturbed and increase monotonicity. Also, test accuracy is improved compared to SGD, LEAP and Lookahead algorithms in some experiments.

## Experiments

Experiments are organised as presets in experiments_pack2.py. Experiments are designed to run comparing optimisers with the identically initialised nets and in the same epochs/minibatches cycle. Five optimisers are included for comparison: cs-dq(netline), so-dz(dz), lookahead, leap and sgd. There are 16 presaved weights for ResNet which we use for experiments and the manual seed for PyTorch dataloader is similarly fixed in range 0 - 15. Optimisers cs-dq and so-dz have built-in lookahead option and different experiments test the optimisers with lookahead on/off. The following experiments are available:

| Experiment Num | Description | Lookahead Option |
| ---- | --------------------- | --- |
| 100  | RESNET18/CIFAR10      | on  |
| 101  | RESNET18/CIFAR10      | off |
| 110  | RESNET18/CIFAR100     | on  |
| 111  | RESNET18/CIFAR100     | off |
| 120  | RESNET34/CIFAR10      | on  |
| 121  | RESNET34/CIFAR10      | off |
| 130  | RESNET34/CIFAR100     | on  |
| 131  | RESNET34/CIFAR100     | off |
| 140  | RESNET18/SVHN         | on  |
| 150  | RESNET18/ImageNet100  | off |
| 151  | RESNET18/TinyImageNet | off |

### How to run experiment
To run the experiment, run the following command in your terminal:

```bash
python experiments_pack2.py <experiment_num> <job_number> <epochs_number> <optimisers_mask> <manual_seed>
```

#### Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `experiment_num` | `number` | **Yes** | *None* | Experiment num from the list above. |
| `job_number` | `string` | No | `0` | String id to use in log-file name as a suffix. |
| `epochs_number` | `number` | No | `50` | Number of epochs in the run. |
| `optimisers_mask` | `number` | No | `31` | Bitmask to select optimisers. The optimisers have the following ids: cs-dq(netline) = 1, so-dz(dz) = 2, lookahead = 4, leap = 8, sgd = 16. The bitmask by bitwise and with the ids excludes the optimisers. For example, optimisers_mask = 15 will exclude sgd. If the bitmask not set, all optimisers are included. |
| `manual_seed` | `number` | No | `-1` | Value from 0 to 15 to use fixed net initialisation and data loader. If not set, seed is not fixed. |

#### Example

Example of command to run experiment 100 with sgd excluded:

```bash
python experiments_pack2.py 100 146 50 15 0
```
