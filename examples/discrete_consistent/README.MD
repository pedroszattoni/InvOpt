
# `discrete_consistent`

>Warning: the `discrete_consistent` function requires `gurobipy`.

## Use case

This function can be used when the FOP is of the form

$$
\min_ {x \in \mathbb{X}(\hat{s})} \ \langle \theta,\phi(\hat{s},x) \rangle ,
$$

the constraint set $\mathbb{X}(\hat{s})$ is discrete, that is, we can list all elements in this set, and when there exists a cost vector $\theta^\star$ consistent with the dataset $\mathcal{D}$, that is, such that $x_ i \in \arg\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ \langle \theta^\star,\phi(\hat{s}_ i,x) \rangle  \ \ \forall i \in [N]$.

## Solution method

Given the a dataset of signal-response data $\mathcal{D} = \\{(\hat{s}_ i, \hat{x}_ i)\\}_ {i=1}^N$,  the `discrete_consistent` function solves the IO problem using the *incenter stratety*

$$
\begin{aligned}
 \min_ {\theta} \quad & \mathcal{R} (\theta - \hat{\theta}) \\ 
 \text{s.t.} \quad &  \langle \theta,\phi(\hat{s}_ i,\hat{x}_ i) - \phi(\hat{s}_ i,x_ i) \rangle + d(\hat{x}_ i,x_ i)\leq 0 \quad\quad \forall x_ i \in \mathbb{X}(\hat{s}_ i), \ \forall i \in [N] \\
 & \theta \in \Theta,
\end{aligned}
$$

where
- $\hat{\theta} \in \mathbb{R}^p$ is an a priory belief or estimate of the true cost vector. *(optional)*
- $\mathcal{R} : \mathbb{R}^p \to \mathbb{R}$ is a regularization function.
- $\phi: \mathbb{S} \times \mathbb{X} \to \mathbb{R}^p$ is the feature mapping, which maps a signal-response pair $(\hat{s},x)$ to a feature vector $\phi(\hat{s},x)$.
- $d : \mathbb{X} \times \mathbb{X} \to \mathbb{R}_+$ is a distance penalization function, which given two responses, returns the distance between them according to some distance metric. *(optional)*
- $\mathbb{X}(\hat{s}_ i)$ is the constraint set of the FOP.
- $\Theta$ is the set used to encode any prior information or assumption we may have on the expert's true cost function, e.g., nonnegativity of the cost vector. *(optional)*

Alternatively, if the distance penalization function $d$  and the priory belief/estimate vector $\hat{\theta}$ are not provided, the `discrete_consistent` function solves the IO problem by solving the *feasibility strategy*

$$
\begin{aligned} \min_ {\theta} \quad & 0 \\
\text{s.t.} \quad & \langle \theta,\phi(\hat{s}_ i,\hat{x}_ i) - \phi(\hat{s}_ i,x_ i) \rangle \leq 0 \quad\quad \forall x_ i \in \mathbb{X}(\hat{s}_ i), \ \forall i \in [N] \\
& \theta \in \Theta, \quad \rVert \theta \rVert = 1,
\end{aligned}
$$

where the norm equality constraint uses the $\ell_ 1$ norm if $\Theta = \\{\theta \in \mathbb{R}^p : \theta \geq 0\\}$, and the $\ell_ \infty$ norm otherwise. This equality constraint must be added to exclude the trivial solution $\theta = 0$. Notice that both the incenter and feasibility optimization programs become infeasible if the dataset is not consistent. For more details on these reformulations, see [Zattoni Scroccaro et al. (2023)](https://arxiv.org/abs/2305.07730). 

## Example: binary linear program with consistent data

>Warning: the example in `binary_LP_consistent_data.py` requires the pacakge [`polytope`](https://github.com/tulip-control/polytope).

In the file `binary_LP_consistent_data.py`, you will find an example usage of the `discrete_consistent` function. For this example, the dataset $\mathcal{D} = \\{(\hat{s}_ i, \hat{x}_ i)\\}_ {i=1}^N$ is generated by solving the binary linear program

$$
\begin{aligned}
\hat{x}_ i \in \arg\min_ {x} \quad &  \langle \theta,x \rangle \\
\text{s.t.} \quad & \hat{A}_ i x \leq \hat{b}_ i \\
& x \in \\{0,1\\}^n,
\end{aligned}
$$

where $\hat{s}_ i = (\hat{A}_ i, \hat{b}_ i)$ and $\phi(\hat{s},x) = x$. The IO problem is solved using the incenter and feasibility strategies from `discrete_consistent`, as well as the circumcenter strategy from [Besbes et al. (2022)](https://arxiv.org/abs/2106.14015). For a detailed description of this example, please see [Zattoni Scroccaro et al. (2023)](https://arxiv.org/abs/2305.07730).