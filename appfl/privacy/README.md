# Differential Privacy (DP) 

## Definition of $\bar{\epsilon}$-DP
A randomized function $\mathcal{A}$ provides $\bar{\epsilon}$-DP if, 
for any two datasets $\mathcal{D}$ and $\mathcal{D}'$ that differ in a single entry and for any set $\mathcal{S}$,

$$
\Big| \ln \Big( \frac{\mathbb{P}( \mathcal{A}(\mathcal{D}) \in \mathcal{S} ) }{\mathbb{P}( \mathcal{A}(\mathcal{D}') \in \mathcal{S} )  } \Big) \Big| \leq  \bar{\epsilon},
$$

where $\mathcal{A}(\mathcal{D})$ (resp. $\mathcal{A}(\mathcal{D'})$) is a randomized output of $\mathcal{A}$ on input $\mathcal{D}$ (resp. $\mathcal{D}'$).

This implies that as $\bar{\epsilon}$ decreases, it becomes hard to distinguish the two dataset $\mathcal{D}$ and $\mathcal{D}'$ by analyzing the randomized output of $\mathcal{A}$.
Here, $\bar{\epsilon}$ is a privacy budget controlled by the users (note: stronger privacy is achieved with a lower $\bar{\epsilon}$). 

## Output Perturbation Method
A popular way of constructing $\mathcal{A}$ that ensures $\bar{\epsilon}$-DP is to add some noises **directly** to the true output $\mathcal{T}(\mathcal{D})$ of a function, namely, 
$$
\mathcal{A}(\mathcal{D}) = \mathcal{T}(\mathcal{D}) + \tilde{\xi},
$$
which is known as the ``output perturbation`` method. 

### Laplace mechanism in the output perturbation
To construct a randomized function $\mathcal{A}$ that ensures $\bar{\epsilon}$-DP based on the output perturbation method, we sample $\tilde{\xi}$ from a Laplace distribution with the following probability densition function:

$$
\text{Lap} (\tilde{\xi}; 0, \bar{\Delta}/\bar{\epsilon} ) = \frac{1}{2(\bar{\Delta}/\bar{\epsilon})} e^{(-\frac{\|\tilde{\xi}\|_1 }{\bar{\Delta}/\bar{\epsilon}})},
$$

where $\bar{\Delta}$ should satisfy 

$$
\bar{\Delta} \geq \| \mathcal{T}(\mathcal{D}) - \mathcal{T}(\mathcal{D}') \|_1, 
$$
for all two datasets $(\mathcal{D}, \mathcal{D}')$ that differ in a single data point.

**Example 1.** How to compute $\bar{\Delta}$ that ensures $\bar{\epsilon}$-DP for every iteration of the fedrated averaging (FedAvg) algorithm?

Recall that a supervised machine learning model is given by
$$
\min_{w} \sum_{i=1}^{I}  \frac{1}{I} f_i(w; x_i, y_i),
$$
where
- $w$ is a model parameter vector,
- $I$ is the total number of data points,
- $x_i$ is the $i$-th data input point,
- $y_i$ is the $i$-th data label point,
- $f_i$ is the loss function.

This is equivalent to 
$$
\min_{w} \sum_{p=1}^{P} \frac{I_p}{I} \frac{1}{I_p} \sum_{i \in \mathcal{I}_p} f_i(w; x_i, y_i),
$$
where
- $P$ is the number of clients,
- $\mathcal{I}_p$ is a set of data points from a client $p$
- $I_p= |\mathcal{I}_p|$ and $I = \sum_{p=1}^P I_p$.

Let $\eta$ be a learning rate (or step size). 
Then, for every iteration $t$ of a gradient-based algorithm, the model parameter is updated by
$$
\begin{aligned}
w^{t+1} & = w^t - \eta \nabla \Big( \sum_{p=1}^{P}  \frac{I_p}{I} \frac{1}{I_p} \sum_{i \in \mathcal{I}_p} f_i(w^t; x_i, y_i) \Big) \\
& = \sum_{p=1}^{P}  \frac{I_p}{I} \Big\{ w^t - \eta  \nabla \Big( \frac{1}{I_p} \sum_{i \in \mathcal{I}_p} f_i(w^t; x_i, y_i) \Big) \Big\}.
\end{aligned}
$$

For every client $p \in [P]$, we define a local model parameter
$$
z^{t+1}_p := w^t - \eta  \nabla \Big( \frac{1}{I_p} \sum_{i \in \mathcal{I}_p} f_i(w^t; x_i, y_i) \Big).
$$
Then the global model parameter can be obtained by
$$
w^{t+1} = \sum_{p=1}^{P}  \frac{I_p}{I} z^{t+1}_p.
$$

This is the popular federated averaging algorithm.

Our goal is to preserve privacy on an every single data point in $\mathcal{D}_p := \{(x_i, y_i)\}_{i \in \mathcal{I}_p}$ for all clients $p \in [P]$.
To this end, for every clients $p \in [P]$ and iteration $t$, a Laplacian noise vector $\tilde{\xi}^t_p$ sampled from $\text{Lap}(\tilde{\xi}^t_p; \bar{\Delta}^t_p/\bar{\epsilon})$ is added to the true output $z^{t+1}_p$, where $\bar{\Delta}^t_p$ should satisfy
$$
\bar{\Delta}^t_p \geq \max_{\mathcal{D}'_p \in \overline{\mathcal{D}}_p} \| z^{t+1}_p(\mathcal{D}_p) - z^{t+1}_p(\mathcal{D}'_p) \|_1,
$$
where $\overline{\mathcal{D}}_p$ is a collection of $\mathcal{D}'_p$ differing a single data point from $\mathcal{D}_p$.

**Case 1.** Consider
$$
\overline{\mathcal{D}}_p := \Big\{ \mathcal{D}_{pj} = \{ (x_i,y_i)_{i \in \mathcal{I}_p \setminus \{j\} } \}   : j \in \mathcal{I}_p \Big\}. 
$$
By this construction, we have
$$
\begin{aligned}
\bar{\Delta}^t_p 
& \geq \max_{j \in \mathcal{I}_p} \| z^{t+1}_p(\mathcal{D}_p) - z^{t+1}_p(\mathcal{D}_{pj}) \|_1 \\
& = \max_{j \in \mathcal{I}_p} \Big\| \Big( w^t - \eta  \frac{1}{I_p} \sum_{i \in \mathcal{I}_p}  \nabla f_i(w^t; x_i, y_i) \Big) - \Big( w^t - \eta  \frac{1}{I_p-1} \sum_{i \in \mathcal{I}_p \setminus \{j\} }  \nabla f_i(w^t; x_i, y_i) \Big) \Big\|_1 \\
& = \max_{j \in \mathcal{I}_p} \eta \Big\| - \frac{1}{I_p} \sum_{i \in \mathcal{I}_p}  \nabla f_i(w^t; x_i, y_i) + \frac{1}{I_p-1} \sum_{i \in \mathcal{I}_p \setminus \{j\} }  \nabla f_i(w^t; x_i, y_i) \Big\|_1 \\
& = \max_{j \in \mathcal{I}_p} \eta \Big\| - \frac{1}{I_p} \nabla f_j(w^t; x_j, y_j) + \Big(\frac{1}{I_p-1} - \frac{1}{I_p} \Big) \sum_{i \in \mathcal{I}_p \setminus \{j\} }  \nabla f_i(w^t; x_i, y_i) \Big\|_1 .
\end{aligned}
$$

**Case 2.** Consider
$$
\overline{\mathcal{D}}_p := \Big\{ \mathcal{D}_{pj} = \{ (x_i,y_i)_{i \in \mathcal{I}_p \setminus \{j\} } \ \cup \ (x'_j, y'_j)  \}   : j \in \mathcal{I}_p \Big\}. 
$$
By this construction, we have
$$
\begin{aligned}
\bar{\Delta}^t_p 
& \geq \max_{j \in \mathcal{I}_p} \| z^{t+1}_p(\mathcal{D}_p) - z^{t+1}_p(\mathcal{D}_{pj}) \|_1 \\
& = \frac{\eta}{I_p} \max_{j \in \mathcal{I}_p}  \Big\| -  \nabla f_j(w^t; x_j, y_j) +  \nabla f_j(w^t; x'_j, y'_j) \Big\|_1.
\end{aligned}
$$
If the gradient is clipped by $C$ (e.g., $\nabla f_i \leftarrow \nabla f_i / \max\{ 1, \| \nabla f_i  \|_1 /C \}$ ), then $\|\nabla f_i \|_1 \leq C$ and 
thus $\bar{\Delta}^t_p = \frac{2C \eta}{I_p}$.

**Example 2.** How to compute $\bar{\Delta}$ that ensures $\bar{\epsilon}$-DP for every iteration of the inexact ADMM (IADMM) algorithm?

The federated learning model can be expressed by
$$
\begin{aligned}
\min_{w, z_1, ... , z_P} \ & \sum_{p=1}^P \Big\{ \frac{1}{I}  \sum_{i \in \mathcal{I}_p}  f_{i} (z_p; x_{i}, y_{i}) \Big\} \\
\text{s.t.} \ & w=z_p, \forall p \in [P],
\end{aligned}
$$
where
- $w$ is a global model parameter,
- $z_p$ is a local model parameter. 

To solve such a model, an ADMM algorithm can be utilized which solves the following sequence of subproblems:

$$
\begin{aligned}
& w^{t+1} \leftarrow \argmin_{w} \ \sum_{p=1}^P \Big( \langle \lambda^t_p, w \rangle + \frac{\rho^t}{2} \|w - z^t_p\|^2 \Big), \\
& z^{t+1}_p \leftarrow \argmin_{z_p} \  \frac{1}{I}  \sum_{i \in \mathcal{I}_p}  f_{i} (z_p; x_{i}, y_{i}) - \langle \lambda^t_p, z_p \rangle +  \frac{\rho^t}{2} \|w^{t+1}-z_p\|^2, \ \forall p \in [P], \\
& \lambda^{t+1}_p \leftarrow  \lambda^{t}_p + \rho^t (w^{t+1}-z^{t+1}_p), \ \forall p \in [P], 
\end{aligned}
$$
where $\lambda_p$ is a dual vector associated with a consensus constraint $w=z_p$ and $\rho$ is an ADMM penalty parameter which should be fine-tuned.


In the inexact ADMM algorithm, the first-taylor approximation is applied on the function yielding
$$
\begin{aligned}
& z^{t+1}_p \leftarrow \argmin_{z_p} \ \Big\langle \frac{1}{I}  \sum_{i \in \mathcal{I}_p}  \nabla f_{i} (z_p^t; x_{i}, y_{i}), z_p \Big\rangle - \langle \lambda^t_p, z_p \rangle +  \frac{\rho^t}{2} \|w^{t+1}-z_p\|^2, \ \forall p \in [P], \\
& z^{t+1}_p = w^{t+1} + \frac{1}{\rho^t} \Big(\lambda^t_p - \frac{1}{I} \sum_{i \in \mathcal{I}_p} \nabla f_i(z_p^t; x_i, y_i) \Big)
\end{aligned}
$$

Within the context of IADMM, $\bar{\Delta}^t_p$ can be computed as follows:
$$
\begin{aligned}
\bar{\Delta}^t_p 
& = \max_{\mathcal{D}'_p \in \overline{\mathcal{D}}_p} \Big\| \Big( w^{t+1} + \frac{1}{\rho^t} \Big(\lambda^t_p - \frac{1}{I} \sum_{i \in \mathcal{I}_p} \nabla f_i(z_p^t; x_i, y_i) \Big) \Big) - \Big( w^{t+1} + \frac{1}{\rho^t} \Big(\lambda^t_p - \frac{1}{I} \sum_{i \in \mathcal{I}_p} \nabla f_i(z_p^t; x'_i, y'_i) \Big) \Big) \Big\|_1 \\
& = \frac{1}{\rho^t I} \max_{i \in \mathcal{I}_p} \Big\|  \nabla f_i(z^t_p; x_i, y_i) - \nabla f_i(z^t_p; x'_i, y'_i) \Big\|_1.
\end{aligned}
$$