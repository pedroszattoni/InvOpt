# `discrete_model_consistent`

$$
\begin{aligned}
 \min_ {\theta} \quad & \mathcal{R} (\theta - \hat{\theta}) \\ 
 \text{s.t.} \quad & \theta^\top \big(\phi(\hat{s}_ i,\hat{x}_ i) - \phi(\hat{s}_ i,x_ i) \big) + d(\hat{x}_ i,x_ i)\leq 0 \quad\quad \forall x_ i \in \mathbb{X}(\hat{s}_ i), \ \forall i \in [N] \\
 & \theta \in \Theta, 
\end{aligned}
$$
where
