# Theoretical Limits of KRONIC with RBF Kernels for Financial Control

This document provides a purely theoretical diagnosis of why the exact continuous-time KRONIC formulation breaks down when applied to the Merton Portfolio problem using standard RBF (Radial Basis Function) kernels, and outlines mathematically sound theoretical solutions.

## 1. The True Objective and the Variance Penalty

The Merton problem seeks to maximize expected terminal utility $\mathbb{E}[U(W_T)]$. For CRRA utility $U(W) = \frac{W^{1-\gamma}}{1-\gamma}$, we have established that the utility acts like an eigenfunction of the diffusion generator.

The exact growth rate of the expected utility is governed by the certainty-equivalent growth rate:
$$ g(\pi, V) = r + \pi(\mu - r) - \frac{\gamma}{2}\pi^2 V $$

The negative quadratic term $-\frac{\gamma}{2}\pi^2 V$ is the fundamental variance penalty. This penalty is the only mechanism that prevents the optimal allocation $\pi^*$ from going to infinity. The true optimal allocation balances the linear drift against this quadratic penalty:
$$ \pi^\* = \frac{\mu - r}{\gamma V} $$

## 2. KRONIC's LP Formulation

KRONIC solves this by learning Koopman eigenfunctions $\psi_i(z)$ from data, representing the utility as a projection $U \approx \sum_i w_i \psi_i$, and then maximizing the explicit expected utility formulation:

$$ \max\pi \mathbb{E}[U(W_T)] \approx \max\pi \sum\_{i=1}^M w_i e^{\lambda_i(\pi) T} \psi_i(W_0, \pi, V_0) $$

Where the eigenvalues $\lambda_i(\pi)$ are estimated either analytically or via numerical directional derivatives of the eigenfunctions:
$$ \lambda_i(\pi) = \frac{L\psi_i(z)}{\psi_i(z)} $$

## 3. The RBF Extrapolation Breakdown

The breakdown occurs in the representation space chosen for $\psi_i(z)$. Using Kernel Extended Dynamic Mode Decomposition (KGEDMD) with an RBF kernel, we represent the eigenfunctions in the span of the data:
$$ \psi*i(z) = \sum*{k=1}^N v\_{i,k} \exp\left(-\frac{\|z - z_k\|^2}{2\sigma^2}\right) $$

where $z = (\log W, \pi, V)$ and $z_k$ are the training samples.

### 3.1 The Boundedness of RBF Derivatives

The generator $L$ applied to an RBF kernel produces terms involving the drift $b(z)$ and diffusion $a(z)a(z)^T$.
However, fundamentally, the RBF kernel and all its spatial derivatives decay exponentially to zero as $\|z - z_k\| \to \infty$.

$$ \lim{\pi \to \infty} k(z, z*k) = 0 $$
$$ \lim*{\pi \to \infty} \nabla*\pi k(z, z_k) = 0 $$
$$ \lim*{\pi \to \infty} \nabla^2\_\pi k(z, z_k) = 0 $$

### 3.2 The Collapse of the Variance Penalty

Because the exact variance penalty $-\frac{\gamma}{2}\pi^2 V$ is a global, unbounded quadratic, it cannot be accurately represented globally by a finite sum of localized, bounded RBF bumps.

Within the training region $[- \pi_{max}, \pi_{max}]$ where data exists, the RBF sum approximates the local curvature correctly. But when the optimizer queries a target allocation $\pi_{\text{target}}$ outside the training distribution:

1. The distance to all training points $\|z_{\text{target}} - z_k\|^2$ becomes very large.
2. The RBF weights collapse towards 0.
3. The estimated eigenvalue $\lambda(\pi)$ flatten out or trend unpredictably, losing the $-\frac{\gamma}{2}\pi^2 V$ structure.
4. The controller "forgets" that high leverage causes massive variance.

Because the risk penalty suddenly disappears outside the support of the training data, the optimizer aggressively exploits this mathematical blindspot, assigning infinite expected utility to extreme allocations (e.g., hitting the artificial bounds of $\pi = 4.0$ or $\pi = 10.0$).

This explains why previous implementations only worked when the search grid for $\pi$ was "cheating" by being highly localized around the known analytical optimum.

### 3.3 The Augmented State and Piecewise Constant Control

A natural question arises: is the choice to treat the control action $\pi$ as a state variable (augmenting the state to $z = (\log W, \pi, V)$) fundamentally flawed because it forces the operator to predict "future $\pi$'s"?

No. It is mathematically perfectly equivalent to the **piecewise constant control assumption** commonly used in numerical schemes and HJB verification theorems.

When we augment the state and generate training data where the action is held constant ($d\pi = 0$), the augmented infinitesimal generator $L_{aug}$ exactly matches the true parameterized generator $L^\pi$:
$$ L*{aug} \psi(W, \pi, V) = \left( L^\pi \right) \psi(W, \cdot, V) $$
The dynamics of $\pi$ itself have exactly zero drift and zero diffusion. The Koopman operator $e^{L*{aug} \Delta t}$ therefore rigorously predicts the expected utility under the exact assumption that we take action $\pi_t$ and hold it mathematically constant over the evaluation interval $[t, t+\Delta t]$.

**Therefore, the augmented state formulation itself is not the flaw.** It is a rigorous, elegant way to encode parameterized control generators without painstakingly deriving an explicit algebraic control-affine structure.

The extrapolation breakdown is entirely a **function approximation failure**. While $L_{aug}$ strictly contains the global $-\frac{\gamma}{2}\pi^2 V$ variance penalty in its algebraic definition, the chosen hypothesis space—a linear combination of localized RBF bumps—is fundamentally incapable of representing this unbounded quadratic operator globally across the $\pi$ dimension. The continuous-time Koopman operator is correct conceptually, but its RBF incarnation collapses beyond the training support.

## 4. Theoretical Solutions

To correct this, we must align the hypothesis space (the Kernel) with the required structural properties of the control problem, or alter the optimization loop to gracefully handle extrapolation limits.

### Solution A: Cumulative Path Signatures

Instead of manually constructing polynomial ad-hoc kernels, we employ Cumulative Path Signatures. The log-signature of the state path $S(Z_{[0,t]})$ inherently provides a set of universal non-linear coordinates spanning polynomial combinations of the state increments.

Definition 1 (Truncated Path Signature)  
For a continuous, bounded variation path $Z: [0, t] \to \mathbb{R}^d$, the truncated path signature of degree $M$ is the collection of iterated integrals:
$$ S^{(M)}(Z)_{[0,t]} = \left( 1, \int_{0}^t dZ*{s_1}, \iint*{0 < s*1 < s_2 < t} dZ*{s*1} \otimes dZ*{s*2}, \dots, \int \dots \int*{0 < s*1 < \dots < s_M < t} dZ*{s*1} \otimes \dots \otimes dZ*{s_M} \right) $$

Proposition 1 (Exact Generator Representation via Signatures)  
_Let the state be $Z = (\log W, \pi, V)$ governed by Itô diffusions $dZ_t = b(Z_t)dt + a(Z_t)dB_t$. A linear functional over the level-2 log-signature of the path perfectly recovers the infinitesimal generator $L$, and thus the quadratic variance penalty $-\frac{\gamma}{2}\pi^2 V$, without requiring localized extrapolation._

Proof:  
By Itô's Lemma, the infinitesimal expected change of any smooth function $f(Z)$ depends on both the first-order drift and the second-order quadratic variation:
$$ \mathbb{E}[df(Z_t)] = \nabla f(Z_t) \cdot \mathbb{E}[dZ_t] + \frac{1}{2} \text{Tr}\left( \nabla^2 f(Z_t) \mathbb{E}[d\langle Z \rangle_t] \right) $$

The first level of the signature $S^{(1)}_{[t, t+\Delta t]} = \int_t^{t+\Delta t} dZ_s \approx b(Z_t) \Delta t + a(Z_t) \Delta B_t$. Taking the expectation recovers the drift $b(Z_t)\Delta t$.
The symmetric part of the second level of the signature is the cross-variation matrix:
$$ \text{Sym}\left(S^{(2)}_{[t, t+\Delta t]}\right) = \frac{1}{2}\left( S^{(1)} \otimes S^{(1)} \right) = \frac{1}{2} d\langle Z \rangle_{t} = \frac{1}{2} a(Z_t) a(Z_t)^T \Delta t $$

Because $\Sigma = a(Z_t) a(Z_t)^T$ contains exactly the term $\pi^2 V$ in the $(W, W)$ component of the diffusion matrix, the variance penalty $-\frac{\gamma}{2}\pi^2 V$ is simply a static linear combination of the coordinates of $S^{(2)}$.
Thus, an inner product $\langle w, S^{(2)} \rangle$ exactly encapsulates the variance penalty structurally. Because $S^{(M)}$ comprises global polynomial terms rather than localized exponential decay (like RBFs), it intrinsically preserves the mathematical property $\lim_{\pi \to \infty} \mathbb{E}[\langle w, S^{(2)}\rangle] \to -\infty$. $\blacksquare$

### Solution B: Online KRONIC Controller (Exploration-Correction)

If we do not want to constrain the controller with artificial bounds or change the kernel, we can treat the extrapolation breakdown as a feature of an Online Learning loop. When the controller queries an allocation $\pi_{extrap}$ that is far outside its training data, it lacks local RBF support for the variance penalty, thus hallucinating a high expected utility.

Definition 2 (Online Koopman Eigenvalue Update)  
Let $G_{00}^{(t)}$ and $G_{10}^{(t)}$ be the empirical Gram matrices of the Koopman generator at time $t$ over $N$ samples. Given a new observed state transition $(Z_t, Z_{t+\Delta t})$ under action $\pi$, leaving features $\psi(Z_t)$ and approximate generator derivative $d\psi(Z_t) = \frac{\psi(Z_{t+\Delta t}) - \psi(Z_t)}{\Delta t}$, the rank-1 update (Recursive Least Squares) to the generator $A^{(t+1)}$ is:
$$ A^{(t+1)} = A^{(t)} + \frac{P^{(t)} \psi(Z*t) (d\psi(Z_t) - A^{(t)}\psi(Z_t))^T}{1 + \psi(Z_t)^T P^{(t)} \psi(Z_t)} $$
where $P^{(t)} = (G*{00}^{(t)})^{-1}$ is the inverse covariance matrix, updated via the Sherman-Morrison formula.

Proposition 2 (Local Suboptimality Correction Bound)  
_Assume the offline KRONIC model hallucinates an extreme false optimum $\pi_{extrap} \gg \pi^_$ such that the prior estimated eigenvalue $\hat{\lambda}_{prior}(\pi*{extrap}) > \lambda(\pi^*)$, due to RBF decay $\nabla^2 k(Z, Z_{extrap}) \approx 0$. If the controller takes action $\pi_{extrap}$ for a single step $\Delta t$, the online rank-1 update strictly decreases the estimated eigenvalue at $\pi_{extrap}$ proportionally to the realized structural variance penalty, guaranteeing rejection of the false optimum in the next step.\_

Proof:  
Let the true instantaneous generator action at the extrapolated state $Z_{extrap}$ be $L\psi$. Because $\pi_{extrap}$ is large, the empirical transition $d\psi_{extrap}$ is highly volatile. By Itô's Lemma:
$$ \mathbb{E}[d\psi_{extrap}] = b(Z*{extrap}) \nabla \psi + \frac{1}{2} \text{Tr}(\Sigma(Z*{extrap}) \nabla^2 \psi) \Delta t $$
Since $\Sigma(Z_{extrap})$ contains $(\pi_{extrap})^2 V$, the variance of the observed transition is massive. 
Because the prior model $A^{(t)}$ suffered from RBF decay, its predicted derivative $A^{(t)}\psi(Z_{extrap}) \approx b(Z_{extrap}) \nabla \psi$, missing the massive negative quadratic term.
The innovation error $e_t = d\psi(Z_t) - A^{(t)}\psi(Z_t)$ will thus be overwhelmingly dominated by the observed empirical variance. The RLS update instantly drags the newly computed local eigenvalue $\hat{\lambda}_{post} = \frac{(A^{(t+1)}\psi)^T \psi}{\| \psi \|^2}$ down strictly towards the true negative realization:
$$ \mathbb{E}[\hat{\lambda}_{post}(\pi_{extrap})] \approx \hat{\lambda}_{prior}(\pi_{extrap}) - \mathcal{O}\left( (\pi*{extrap})^2 V \right) $$
Because the variance penalty is quadratic in $\pi$, a single realized finite step $\Delta t$ at an extreme leverage produces enough empirical squared-error to overwhelm the prior. The eigenvalue collapses to negative, and the LP optimizer strictly rejects $\pi*{extrap}$ for $t+1$. $\blacksquare$

An online controller naturally bounds itself: any dangerous extrapolation immediately generates the massive data needed to correct the model locally. Over time, the controller organically and safely maps out the boundaries of the safely traversable envelope.
