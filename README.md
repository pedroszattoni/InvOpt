
# InvOpt: Inverse Optimization with Python

InvOpt is a Python package for solving Inverse Optimization (IO) problems. In IO problems, our goal is to model the behavior of an expert agent, which given an exogenous signal, returns a response action. The underlying assumption of IO is that to compute its response, the expert agent solves an optimization problem parametric in the exogenous signal. We assume to know the constraints imposed on the expert, but not its cost function. Therefore, our goal is to model the cost function being optimized by the expert, using examples of exogenous signals and corresponding expert response actions. More concretely, given a dataset $\mathcal{D} = \\{(\hat{s}_ i, \hat{x}_ i)\\}_ {i=1}^N$ of exogenous signals $\hat{s}_ i$ and the respective expert's response $\hat{x}_ i$, our goal is to find a cost vector $\theta \in \mathbb{R}^p$ such that the **Forward Optimization Problem (FOP)** with feature mapping $\phi$
$$x_i \in \arg\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ \theta^\top \phi(\hat{s}_ i,x)$$ reproduces (or in some sense approximates) the expert's action $\hat{x}_ i$. For a more detailed description of the IO problem and its modelling, please refer to [Zattoni Scroccaro et al. (2023)](https://arxiv.org/abs/2305.07730) and the references therein. 

## Installation

```bash
pip install invopt
```
InvOpt depends on NumPy. Moreover, some of its functions also depend on gurobipy or cvxpy. You can get a free academic license for Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Usage and examples

The following functions are available in the InvOpt package to solving IO problems:

- [`discrete_model_consistent`](https://github.com/pedroszattoni/invopt/tree/main/examples/discrete_model_consistent): for FOPs with dicrete decision spaces (e.g., binary), and when the dataset is consistent with some cost vector. Can be used to check if the data is consistent.
- [`discrete_model`](https://github.com/pedroszattoni/invopt/tree/main/examples/discrete_model): for FOPs with dicrete decision spaces (e.g., binary).
- [`MIP_linear`](https://github.com/pedroszattoni/invopt/tree/main/examples/MIP_linear): for FOPs with mixed-integer decision spaces and cost functions linear w.r.t. the continuous part of the decision variable.
- [`MIP_quadratic`](https://github.com/pedroszattoni/invopt/tree/main/examples/MIP_quadratic): for FOPs with mixed-integer decision spaces and cost functions quadratic w.r.t. the continuous part of the decision variable.
- [`FOM`](https://github.com/pedroszattoni/invopt/tree/main/examples/FOM): for general FOPs. Solves IO problem approximately using first-order methods.

## Contributing

Contributions, pull requests and suggestions are very much welcome. The  [TODO](https://github.com/pedroszattoni/invopt/blob/main/TODO.txt) file contains a number of ideas to possibly improve the InvOpt package.

## Citing
If you use InvOpt for research, please cite our accompanying paper:

```bibtex
@article{zattoniscroccaro2023learning,
  title={Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality Loss, and Algorithms},
  author={Zattoni Scroccaro, Pedro and Atasoy, Bilge and Mohajerin Esfahani, Peyman},
  journal={https://arxiv.org/abs/2305.07730},
  year={2023}
}
```