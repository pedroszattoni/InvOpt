# `discrete_model_consistent`

Given the a dataset of signal-response data $\mathcal{D} = \{(\hat{s}_ i, \hat{x}_ i)\}_ {i=1}^N$, the FOP
$$\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ \theta^\top \phi(\hat{s}_ i,x),$$ and assuming there exists a cost vector $\theta^\star$ consistent with the dataset, that is, such that $x_ i \in \arg\min_ {x \in \mathbb{X}(\hat{s}_ i)} \ {\theta^\star}^\top \phi(\hat{s}_ i,x)$. The `discrete_model_consistent` function solves the IO problem (by default) using the *incenter stratety*
$$
\begin{aligned}
 \min_ {\theta} \quad & \mathcal{R} (\theta - \hat{\theta}) \\ 
 \text{s.t.} \quad & \theta^\top \big(\phi(\hat{s}_ i,\hat{x}_ i) - \phi(\hat{s}_ i,x_ i) \big) + d(\hat{x}_ i,x_ i)\leq 0 \quad\quad \forall x_ i \in \mathbb{X}(\hat{s}_ i), \ \forall i \in [N] \\
 & \theta \in \Theta, 
\end{aligned}
$$

where
