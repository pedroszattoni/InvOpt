# InvOpt: Inverse Optimization in Python

InvOpt is a Python package for solving Inverse Optimization (IO) problems. In IO problems, our goal is to model the behavior of an expert agent, which given an exogenous signal, returns a response action. The underlying assumption of IO is that to compute its response, the expert agent solves an optimization problem parametric in the exogenous signal. We assume to know the constraints imposed on the expert, but not its cost function. Therefore, our goal is to model the cost function being optimized by the expert, using examples of exogenous signals and corresponding expert response actions. More concretely, given a dataset $\mathcal{D} = \{(\hat{s}_ i, \hat{x}_ i)\}_ {i=1}^N$ of exogenous signals $\hat{s}_ i$ and the respective expert's response $\hat{x}_ i$, our goal is to find a cost vector $\theta \in \mathbb{R}^p$ such that the **Forward Optimization Problem (FOP)** with feature mapping $\phi$
$$x_i \in \arg\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ \theta^\top \phi(\hat{s}_ i,x)$$ reproduces (or in some sense approximates) the expert's action $\hat{x}_ i$. For a more detailed description of the IO problem and its modelling, please refer to [Zattoni Scroccaro et al. (2023)](https://arxiv.org/abs/0000.00000) and the references therein. 

## Installation

```bash
pip install invopt
```
InvOpt depends on NumPy. Moreover, some of its functions also depend on gurobipy or cvxpy. You can get a free academic license for Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Usage & examples

The folder [examples](https://github.com/pedroszattoni/invopt/tree/main/examples) contains descriptions of the functions available in the InvOpt package, as well as multiple examples.

## Contributing

Contributions, pull requests and suggestions are very much welcome. The  [TODO](https://github.com/pedroszattoni/invopt/blob/main/TODO.txt) file contains a number of ideas to possibly improve the InvOpt package.

## Citing
If you use InvOpt for research, please cite our accompanying paper:

```bibtex
@article{zattoniscroccaro2023learning,
  title={Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality Loss, and Algorithms},
  author={Zattoni Scroccaro, Pedro and Atasoy, Bilge and Mohajerin Esfahani, Peyman},
  journal={arXiv preprint arXiv:0000.00000},
  year={2023}
}
```