# Lattice QFT with continuous normalizing flows

This repository contains an implementation of continuous normalizing flows for scalar quantum field theory using 
[JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku),
as introduced in the paper [Learning Lattice Quantum Field Theories with Equivariant Continuous Flows](https://arxiv.org/abs/2207.00283).
Specifically, it focuses on ϕ⁴ theory as an example.

![ODE flow example](flow_example.gif)

## Installation and dependencies
The code can be installed as a package via `pip install .` from
within the root directory of this project.
However, to run the code with GPU support, JAX should be installed first following the instructions [here](https://github.com/google/jax#installation).

To run the example scripts and the notebook below,
the additional packages [hydra](https://hydra.cc/) and [matplotlib](https://matplotlib.org/) are required
(install for example with `pip install hydra-core matplotlib`).

## Examples

### Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathisgerdes/continuous-flow-lft/blob/master/notebooks/train-and-mcmc.ipynb)
A step-by-step jupyter notebook with further explanations can be found in `notebooks/train-and-mcmc.ipynb`.
It contains an example of training the normalizing flow and using it to generate samples with a Metropolis-Hastings MCMC step.

### Scripts
Two scripts for training networks as used in the paper are provided: `example_single` and `example_conditional`.
These can be configured by modifying or adding to the configuration files in the folder `configs/`.

Note that training can be slow when running on the CPU.
Especially for a smaller lattice the batch size can be reduced while still yielding good results:
`python example_single.py ++live_plotting=true ++batch_size=64`

## Reference
If you find our work useful, please cite
```
@article{gerdes2022,
    author = "Gerdes, Mathis and de Haan, Pim and Rainone, Corrado and Bondesan, Roberto and Cheng, Miranda C. N.",
    title = "{Learning Lattice Quantum Field Theories with Equivariant Continuous Flows}",
    eprint = "2207.00283",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "7",
    year = "2022"
}
```
