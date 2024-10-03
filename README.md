
# InvOpt: Inverse Optimization with Python

InvOpt is an open-source Python package for solving Inverse Optimization (IO) problems. In IO problems, our goal is to model the behavior of an expert agent, which given an exogenous signal, returns a response action. The underlying assumption of IO is that to compute its response, the expert agent solves an optimization problem parametric in an exogenous signal. We assume to know the constraints imposed on the expert, but not its cost function. Therefore, using examples of exogenous signals and corresponding expert response actions, our goal is to model the cost function being optimized by the expert. More concretely, given a dataset $\mathcal{D} = \\{(\hat{s}_ i, \hat{x}_ i)\\}_ {i=1}^N$ of exogenous signals $\hat{s}_ i$ and the respective expert's response $\hat{x}_ i$, feature mapping $\phi$, our goal is to find a cost vector $\theta \in \mathbb{R}^p$ such that a minimizer $x_ i$ of the **Forward Optimization Problem (FOP)**

$$
x_i \in \arg\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ \langle \theta,\phi(\hat{s}_ i,x) \rangle
$$

reproduces (or in some sense approximates) the expert's action $\hat{x}_ i$. For a more detailed description of IO problems and their modeling, please refer to [Zattoni Scroccaro et al. (2024)](https://pubsonline.informs.org/doi/abs/10.1287/opre.2023.0254) and the references therein. 

## Installation

```bash
pip install invopt
```
InvOpt depends on `numpy`. Moreover, some of its functions also depend on `gurobipy` or `cvxpy`. You can get a free academic license for Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Usage and examples

The following functions are available in the InvOpt package:

- [`discrete_consistent`](https://github.com/pedroszattoni/invopt/tree/main/examples/discrete_consistent): for FOPs with discrete decision spaces (e.g., binary), and when the dataset is consistent with some cost vector. Can be used to check if the data is consistent.
- [`discrete`](https://github.com/pedroszattoni/invopt/tree/main/examples/discrete): for FOPs with dicrete decision spaces (e.g., binary).
- [`continuous_linear`](https://github.com/pedroszattoni/invopt/tree/main/examples/continuous_linear): for continuous, linear FOPs.
- [`continuous_quadratic`](https://github.com/pedroszattoni/invopt/tree/main/examples/continuous_quadratic): for continuous, quadratic FOPs.
- [`mixed_integer_linear`](https://github.com/pedroszattoni/invopt/tree/main/examples/mixed_integer_linear): for FOPs with mixed-integer decision spaces and cost functions linear w.r.t. the continuous part of the decision variable.
- [`mixed_integer_quadratic`](https://github.com/pedroszattoni/invopt/tree/main/examples/mixed_integer_quadratic): for FOPs with mixed-integer decision spaces and cost functions quadratic w.r.t. the continuous part of the decision variable.
- [`FOM`](https://github.com/pedroszattoni/invopt/tree/main/examples/FOM): for general FOPs. Solves IO problem approximately using first-order methods.

## Contributing

Contributions, pull requests and suggestions are very much welcome. The  [TODO](https://github.com/pedroszattoni/invopt/blob/main/TODO.txt) file contains some ideas to possibly improve the InvOpt package.

## Citing
If you use InvOpt for research, please cite our accompanying paper:

```bibtex
@article{zattoniscroccaro2024learning,
  title={Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality Loss, and Algorithms},
  author={Zattoni Scroccaro, Pedro and Atasoy, Bilge and Mohajerin Esfahani, Peyman},
  journal={Operations Research},
  year={2024}
}
```
