# Homework 4

## Multivariate Fisher Kernel

The Fisher score is defined as 
$$
\phi(x)=\left.\nabla_{\theta} \ln p_{\theta}(x)\right|_{\theta=\theta_{0}}.
$$
Meaning $\phi: \mathbb R^d \rightarrow \mathbb R^k$. Next define the Fisher information as
$$
I=E_{p_{\theta_{0}}}\left[\phi(X) \phi^{T}(X)\right],
$$
implying $I \in \mathbb R^{k \times k}$, with $I = I^T$ and $v^TIv \ge 0$. 

1. Verify that $K$ is symmetric.

First, remark that if $A=A^T$ and $A^{-1}$ exists, then $A^{{-1}^T} = A^{-1}$. This is made obvious by the following manipulations,
$$
AA^{-1} = A^{-1}A= A^T A^{-1} = (A^TA^{-1})^T = A^{{-1}^T}A= I.
$$
Next, 
$$
\begin{aligned}
K(x, y) &= \phi^T(x) I^{-1} \phi(y)  \\
&= \left( \phi^T(x) I^{-1} \phi(y)\right)^T \\ 
&=\phi^T(y) I^{{-1}^T} \phi(x).
\end{aligned}
$$
Knowing $I^{{-1}^T} = I^{-1}$,
$$
\begin{aligned}
K(x, y) &= \phi^T(y) I^{{-1}^T} \phi(x) \\
&= K(y, x),
\end{aligned}
$$
satisfying symmetry.

2. Verify that $K$ is positive definite. 

A kernel is positive definite if the following inequality holds,
$$
\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j k(x_i, x_j) \ge 0.
$$
By substitution,
$$
\begin{aligned}
\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j k(x_i, x_j) &= \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j \phi^T(x_i) I^{-1} \phi(x_j) \\
&= \sum_{i=1}^{n} \sum_{j=1}^{n} (\alpha_i \phi(x_i))^T I^{-1} \alpha_j \phi(x_j) \\
&= \left( \sum_{i=1}^{n} \alpha_i \phi(x_i) \right)^T I^{-1} \left(  \sum_{j=1}^{n} \alpha_j \phi(x_j) \right) \\
\end{aligned}
$$
As $I^{-1}$ is also positive definite,
$$
\begin{aligned}
\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j k(x_i, x_j) &= v^T I^{-1} v \\
&\ge 0
\end{aligned}
$$

3. Consider the following multivariate Normal model with given invertible covariance matrix $\Lambda^{-1}$:
   $$
   p_{\theta}(x)=(2 \pi)^{-d / 2}(\operatorname{det} \Lambda)^{1 / 2} \exp \left(-\frac{1}{2}(x-\theta)^{T} \Lambda(x-\theta)\right)
   $$
   Show that
   $$
   \phi(x)=\Lambda\left(x-\theta_{0}\right).
   $$
   Evaluate the gradient
   $$
   \begin{aligned}
   &\nabla_\theta \ln \left[(2 \pi)^{-d / 2}(\operatorname{det} \Lambda)^{1 / 2} \exp \left(-\frac{1}{2}(x-\theta)^{T} \Lambda(x-\theta)\right)\right] \\
   &= \nabla_\theta\left[ -\frac{d}{2} \ln \left( 2\pi \right) + \frac{1}{2} \ln \left( \text{det } \Lambda\right) -\frac{1}{2} (x-\theta)^{T} \Lambda(x-\theta) \right] \\
   &= -\frac{2}{2} \Lambda (x-\theta)(-1) \\
   &= \Lambda(x-\theta).
   \end{aligned}
   $$
   At $\theta = \theta_0$,
   $$
   \phi(x) = \Lambda (x - \theta_0).
   $$

   4. Compute the Fisher information matrix $I$ for this model

   $$
   \begin{aligned} 
   E\left[ \phi(X) \phi^T(X)\right] &= E\left[ \Lambda(X-\theta)(X-\theta)^T \Lambda^T \right] \\
   &= E\left[ \Lambda(X-\theta)(X-\theta)^T \Lambda \right] \\
   &= \Lambda E\left[ (X-\theta)(X-\theta)^T \right] \Lambda\\
   &= \Lambda^3.
   \end{aligned}
   $$

   5. Compute the Fisher kernel
      $$
      \left( \Lambda (x - \theta_0) \right)^T \Lambda^{-1} \Lambda^{-1} \Lambda^{-1} \left( \Lambda (x-\theta_0) \right) = (x-\theta_0)^T \Lambda^{-1} (x-\theta_0).
      $$
      

## Optimal Ordering

Consider the set
$$
\mathcal{D}=\left\{\left(x_{i}, y_{i}\right), 1 \leq i \leq n\right\}
$$


Define the following subsets of indices
$$
\begin{aligned} I_{-} &=\left\{i, 1 \leq i \leq n, y_{i}=-1\right\} \\ I_{+} &=\left\{i, 1 \leq i \leq n, y_{i}=+1\right\} \end{aligned}
$$
Let $n_-$ and $n_+$ denote the order of the subsets respectively. It is desired to construct a function that assigns larger values in the positive data class than the negative class while being smooth in some sense.

Thus, the following functional is defined,
$$
J_{0}(f)=\frac{1}{n_{-} n_{+}} \sum_{i \in I_{-}} \sum_{j \in I_{+}} \mathbb{I}_{f\left(x_{i}\right)>f\left(x_{j}\right)}+\lambda\|f\|_{H}^{2}
$$
To make the functional convex, instead minimize
$$
J(f)=\frac{1}{n_{-} n_{+}} \sum_{i \in I_{-}} \sum_{j \in I_{+}}\left(1-\left(f\left(x_{j}\right)-f\left(x_{i}\right)\right)\right)+\lambda\|f\|_{H}^{2}
$$

1. Show that the minimum is achieved for a function $f$ index by a parameter $\alpha \in \mathbb R^d$ and of the form

$$
f(x)=\sum_{i=1}^{n} \alpha_{i} K\left(x_{i}, x\right)
$$

without using the representer theorem.

Recall from the lest homework:

It is possible to express a function as 
$$
g=g_\alpha + g_\perp,
$$
where 
$$
g_\alpha \in V =\left\{ g_\alpha (x) = \sum_{i=1}^{n} \alpha_i k(x, x_i); \alpha \in \mathbb R^n \right\}
$$
and
$$
\langle g_\perp, h \rangle = 0, \forall h \in V
$$
Now, consider the difference between the functionals,
$$
\begin{aligned} 
J(g, \theta) - J(g_\alpha, \theta) &= \lambda \left( ||g||_{\mathcal H}^2 - ||g_\alpha||_{\mathcal H}^2 \right) \\
&=  \lambda \left( ||g_\perp||_{\mathcal H}^2 + ||g_\alpha||_{\mathcal H}^2 - ||g_\alpha||_{\mathcal H}^2 \right) \\
&= \lambda ||g_\perp||_{\mathcal H}^2
\end{aligned}
$$
Meaning,
$$
J(g, \theta) \ge J(g_\alpha, \theta)
$$
So, the  function $g_\alpha$ that minimizes $J(g_\alpha, \theta)$ also minimizes $J(g, \theta)$.

From above, it is sufficient to show that
$$
J(f) - J(f_\alpha) \ge 0
$$
Meaning,
$$
J(f)-J(f_\alpha)=\frac{1}{n_{-} n_{+}} \sum_{i \in I_{-}} \sum_{j \in I_{+}}\left(-f_\perp\left(x_{j}\right)+f_\perp\left(x_{i}\right)\right)+\lambda\|f_\perp\|_{H}^{2}
$$

However,
$$
f_\perp(x_i) =\langle f_\perp, k(., x_i) \rangle_{\mathcal H} =0
$$
So,
$$
J(f)-J(f_\alpha) = \lambda ||f_\perp||_{\mathcal H}^2 \ge 0.
$$


2. Rewrite $J(f)$ as a functional $J(\alpha)$ using the notation $K$ for the $(n,n)$ matrix $K_{ij} = K(x_i, x_j)$ and $K_i$ for the $i^{th}$ column of $K$.

Matrix Multiplication Versions

a. Dot or Inner Product Form
$$
C=A B=\underbrace{\left[\begin{array}{c}{A_{1, i}} \\ {\vdots} \\ {A_{M, i}}\end{array}\right]}_{M \times K} \underbrace{\left[\begin{array}{cc}{B_{.1}} & {\ldots B_{i N}}\end{array}\right]}_{K \times N}=\underbrace{\left[\begin{array}{ccc}{A_{1, B_{i, 1}}} & {\ldots} & {A_{1, B_{i, N}}} \\ {\vdots} & {} & {} \\ {A_{M, B, 1}} & {\ldots} & {A_{M, B},_{N}}\end{array}\right]}_{M \times N}
$$
b. Column-wise Accumulation
$$
\begin{aligned} C=A B=\left[\begin{array}{lll}{A_{; 1}} & {\ldots} & {A_{:, K}}\end{array}\right]\left[\begin{array}{ccc}{B_{1,1}} & {\ldots} & {B_{1, N}} \\ {\vdots} & {} & {} \\ {B_{K, 1}} & {\ldots} & {B_{K, N}}\end{array}\right] \\ \Longrightarrow C_{:, n}=\left[\begin{array}{ll}{A_{:, 1}} & {\ldots} & {A_{:, K}}\end{array}\right]\left[\begin{array}{c}{B_{1, n}} \\ {\vdots} \\ {B_{K, n}}\end{array}\right]\end{aligned}
$$
c. Matrix-Vector Product
$$
\begin{aligned} C=A B=\underbrace{A}_{M \times K} \underbrace{\left[B_{; 1} \ldots B_{; N}\right]}_{K \times N}=\underbrace{\left[A B_{; 1} \ldots A B_{i, N}\right]}_{M \times N} \\ \Longrightarrow C_{:, j}\end{aligned}
$$
d. Sum of Outer Products
$$
C=A B=\underbrace{\left[\begin{array}{lll}{A_{: 1}} & {\ldots} & {A_{: K}}\end{array}\right]}_{M \times K} \underbrace{\left[\begin{array}{c}{B_{1 ;}} \\ {\vdots} \\ {B_{K}}\end{array}\right]}_{K \times N}
$$

$$
J(\alpha)=\frac{1}{n_{-} n_{+}} \sum_{i \in I_{-}} \sum_{j \in I_{+}}\left(1-\left(\sum_{k=1}^{n} \alpha_{k} K\left(x_{k}, x_j\right)-\sum_{k=1}^{n} \alpha_{k} K\left(x_{k}, x_i\right)\right)\right)+\lambda \alpha^T K \alpha
$$

Or,
$$
J(\alpha) = 1 - \left( \frac{1}{n_+}\sum_{j \in I_{+}} K_j\alpha - \frac{1}{n_-} \sum_{i \in I_{-}} K_i \alpha \right) + \lambda \alpha^T K \alpha
$$

3. Simplify the expression further, using,

$$
\begin{aligned} K_{-} &=\frac{1}{n_{-}} \sum_{i \in I_{-}} K_{i} \\ K_{+} &=\frac{1}{n_{+}} \sum_{i \in I_{+}} K_{i} \end{aligned}
$$

Thus,
$$
J(\alpha) = 1-\left( K_+ \alpha - K_-\alpha\right) + \lambda \alpha^T K \alpha
$$

4. Compute $\nabla_\alpha J(\alpha)$

$$
\nabla_\alpha J(\alpha) = -(K_+^T - K_-^T) + 2 \lambda K \alpha
$$

Thus, for $\nabla_\alpha J(\alpha) = 0$,
$$
\alpha = \frac{1}{2 \lambda}K^{-1} (K_+^T - K_-^T).
$$

5. For the linear kernel $K(x, y) = x^T y$, compute the minimizer $f^*(x)$. 

$$
f^*(x) =\sum_{i=1}^{n} (\alpha_i x_i)^Tx = \alpha^T X x
$$

$$
K = X^TX
$$

$$
\alpha = \frac{1}{2 \lambda} (XX^T)^{-1} (X_+ - X_-)
$$

$$
f^*(x) = \frac{1}{2 \lambda} (X_+ - X_-)^T ((XX^T)^{-1})^T X x
$$

