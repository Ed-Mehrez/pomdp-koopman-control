# Fractional Methods for Stochastic Optimal Control — Reference Report

> **⚠️ SCOPE NOTE**: This is **forward-looking documentation for Stage 5+ planning and the eventual thesis chapter writeup**. It is **NOT part of the Stage 4 v2 critical path**. Codex does not need to read this for v2 implementation. The v2 spec is `docs/derivation_omm_sdre_v2.md`, which is intentionally Markov-only and does not depend on any fSDE / rough vol / signature machinery. Read this document only when (a) doing the `docs/fsde_audit_task.md` audit, (b) planning Stage 5+ work that involves non-Markov dynamics, or (c) writing the related-work section of the thesis chapter.

**Status**: Living document. Initial draft 2026-04-08 (Claude, with web search + sibling repo synthesis).
**Purpose**: Comprehensive reference report on (a) the empirical case for / against rough volatility, (b) the theoretical machinery for stochastic optimal control under fractional / non-Markovian dynamics, (c) practical learning algorithms for high-dimensional fractional control problems, (d) connections to existing sibling repos (`fsde-identifiability`, `fSDE_video_gen`) and to the user's existing paper-in-progress on Koopman spectral roughness estimation.
**Audience**: Future-self, codex when running the audit task, anyone planning Stage 5+ OMM work involving rough or path-dependent dynamics. **NOT for the v2 implementation pass.**
**Companion docs**: `docs/plan_omm_research.md` (the main OMM research plan), `docs/derivation_omm_sdre_v2.md` (the v2 derivation note — Markov-only), `docs/fsde_audit_task.md` (the fSDE audit task spec — has the "intended usage" context for the material in this references doc).

---

## 0. Executive summary

1. **The "rough volatility" question is genuinely unsettled.** The Gatheral-Jaisson-Rosenbaum (2018) "volatility is rough" finding is influential but contested by Cont & Das (2022) "rough volatility: fact or artefact?" and several 2024-2025 papers showing rough Bergomi underperforms simpler Markovian models on parts of the SPX surface. **The right methodological move is to be agnostic to which camp is correct.**

2. **The fSDE control literature has three main mathematical machineries.** In rough order of theoretical age and practical traction:
   - **Wick / white noise / Hida calculus** (Biagini-Hu-Øksendal-Zhang 2008 book): theoretically deep, only fully developed for H > 1/2, considered unsatisfactory for finance because it gives "no arbitrage" via Wick products in a way that doesn't match real markets.
   - **Functional Itô calculus** (Dupire 2009, Cont-Fournié 2010, Ekren-Touzi-Zhang 2014, 2016): rigorous treatment of path-dependent functionals, gives a path-dependent HJB. Works for any H (including rough), no Markov assumption needed.
   - **Stochastic maximum principle for fBm** (Hu-Øksendal 2003, Buckdahn-Ma 2008, Han et al. 2019, recent SIAM J Control 2024): BSDEs for the adjoint process, Malliavin / Doss-Sussmann transformations, partially observed extensions.

3. **For practical high-dimensional learning**, the signature-based approach has emerged as the dominant pragmatic choice. Key references: Cuchiero-Gazzani-Möller signature volatility models (2024), Bayer-Pelizzari signature-kernel American options under rough vol (2025), Bonesini et al. "Hedging with memory" (2025), the rough kernel hedging paper (2025). All explicitly target rough/non-Markovian settings, all are compute-efficient relative to deep RL. **This is what we should use.**

4. **The user's existing sibling repos already provide most of the needed machinery.**
   - `fsde-identifiability` has nonparametric (μ, σ, H) estimation via signatures + Koopman, with a JMLR-target paper outline.
   - `fSDE_video_gen` has Paper 2 specifically on fSDE generative models for time series, with the "reverse-time fSDE" (path-dependent) derivation as a stated target.
   - Paper 6a (in `fsde-identifiability/docs/spectral_roughness_theory.md`) takes an explicit Koopman-spectral position on the rough vol debate.

5. **For OMM specifically**, we should treat fSDEs as the *Stage 5+ test regime* where the methodology demonstrates its value (because Bergault-Guéant doesn't apply), not as a Stage 4 v2 substitute for Heston (where we have a closed-form validation target).

---

## 1. The empirical rough volatility debate

Before deciding to pursue fSDEs over classical stochastic vol, we should be honest that the empirical evidence is mixed.

### 1.1 The "volatility is rough" camp

**Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018).** *Volatility is rough.* [Quantitative Finance 18(6), 933-949](https://www.tandfonline.com/doi/abs/10.1080/14697688.2017.1393551). [arXiv:1410.3394](https://arxiv.org/abs/1410.3394). [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2509457).

The foundational paper. Key finding: log-volatility behaves essentially as a fractional Brownian motion with **Hurst exponent `H ≈ 0.1`** at any reasonable time scale, across multiple equity indices and decades of data. Adopts the **Rough Fractional Stochastic Volatility (RFSV)** model (Comte-Renault with `H < 1/2`). Notes that classical statistical procedures detect long memory in RFSV-generated data even though the model itself has no long memory — a methodological warning.

**Bayer, C., Friz, P. & Gatheral, J. (2016).** *Pricing under rough volatility.* [Quantitative Finance 16(6), 887-904](https://www.tandfonline.com/doi/abs/10.1080/14697688.2015.1099717). [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754).

Introduces the **rough Bergomi (rBergomi)** model, a simple one-factor rough vol specification that fits the SPX implied vol surface remarkably well with only three parameters. Cited as the canonical first demonstration that rough vol models are *practical* for option pricing.

**El Euch, O. & Rosenbaum, M. (2019).** *The characteristic function of rough Heston models.* Mathematical Finance 29(1), 3-38.

Derives the characteristic function for the rough Heston model, enabling Fourier-based pricing. Important because rough Heston is the natural rough generalization of the classical Heston model.

**Bondi, A., et al. (2024).** *The rough Hawkes Heston stochastic volatility model.* [Mathematical Finance, Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12432). [arXiv:2210.12393](https://arxiv.org/pdf/2210.12393).

Most recent extension: rough Heston with self-exciting Hawkes jumps in volatility. State of the art for capturing both the rough vol stylized fact and volatility clustering / jump dynamics.

**Bayer, C., Friz, P. K. & Gatheral, J. (2023).** *Rough Volatility* (book). SIAM Mathematical Modeling and Computation series. [SIAM](https://epubs.siam.org/doi/book/10.1137/1.9781611977783).

The standard textbook reference. Comprehensive treatment of rough vol from theory to numerics. Cite this as the field's textbook.

### 1.2 The skeptics

**Cont, R. & Das, P. (2022).** *Rough volatility: fact or artefact?* [Sankhya B (2024), 86, 191-223](https://arxiv.org/abs/2203.13820). Also at [arXiv:2203.13820](https://arxiv-export3.library.cornell.edu/abs/2203.13820v2).

**This is the most important skeptical paper and we should cite it.** Cont and Das introduce a non-parametric (model-free) estimator of path roughness based on **normalized p-th variation along sequences of partitions**. Their key finding: even when the *true* spot volatility is a standard Brownian motion (`H = 0.5`), the *realized* volatility estimated from price observations exhibits apparent `H < 0.5` due to microstructure noise. **Their conclusion**: the "roughness" detected by Gatheral-Jaisson-Rosenbaum may be an artifact of the estimator, not a property of the underlying volatility process.

**Fukasawa, M. (2021).** Various papers on estimator dependence in rough vol detection. The skeptical view shared by Fukasawa is that different roughness estimators give answers in `H ∈ [0.1, 0.4]`, suggesting the "truth" is unstable.

**Bayer, C. et al. (2024).** *Detecting rough volatility: a filtering approach.* [Quantitative Finance](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2399284). Recent paper that addresses the Cont-Das critique with a filtering-based detector.

**Delemotte, J., De Marco, S. & Segonne, F. (2024).** *Volatility models in practice: Rough, Path-dependent or Markovian?* [arXiv:2401.03345](https://arxiv.org/html/2401.03345v1). Recent comparison: rough Bergomi underperforms a one-factor *Markovian* Bergomi with the same number of parameters on parts of the SPX surface. Concludes that "rough volatility models are inconsistent with the global shape of SPX smiles" — suggests path-dependent Markovian models may be a better practical compromise.

**Survey reference**: Bayer, C., Friz, P. K. & Pelizzari, L. (2025). *A Survey of Rough Volatility.* International Journal of Theoretical and Applied Finance. [Worldscientific](https://www.worldscientific.com/doi/10.1142/S0219024925300021). Bank of Japan IMES discussion paper version: [PDF](https://www.imes.boj.or.jp/research/papers/english/24-E-06.pdf).

**Independent review** (informal): [Volatility: Rough or Not? A Short Review · Chase the Devil](https://chasethedevil.github.io/post/rough-volatility-or-not-a-review/) — useful informal summary of where the debate stands.

### 1.3 The user's existing position: Koopman spectral estimation as the arbiter

The sibling repo `fsde-identifiability` contains `docs/spectral_roughness_theory.md` (Paper 6a, in progress), which proposes that **Koopman generator spectral decay can settle the rough vol debate** by providing a model-free roughness estimator that doesn't suffer from the microstructure-noise contamination issue Cont & Das raise.

Key claim from the Paper 6a doc:

> "Critical Distinction: Most studies use **derivatives market data** (options implied vol surface, high-frequency options trades, VIX futures). Our Approach: Estimate roughness from **underlying asset prices** directly. ... Why Spectral Koopman Settles the Debate: 1) Model-Free: No assumption that true process is fBm; 2) Noise Robustness: Eigenvalue spectrum smooths microstructure noise; 3) Testable Hypothesis: H1 (Gatheral) → H ≈ 0.1-0.2 vs H2 (Cont) → H ≈ 0.3-0.4 vs H3 (Hybrid) → different H for price vs vol."

The main result claimed in that doc: for fBm with Hurst `H`, the eigenvalues `{λ_k}` of the Koopman generator satisfy `|λ_k| ~ k^{-(2H+1)}` as `k → ∞`. This gives a direct read-off of `H` from observed eigenvalue decay, independent of microstructure noise (which affects only the high-frequency tail of the spectrum, leaving the low eigenvalues that determine `H` undisturbed).

**For our OMM work**: this means we should not commit to a position in the rough-vs-classical debate. The methodology should work on both. Paper 6a's spectral estimator is the natural way to characterize the dynamics empirically without committing to a parametric model.

### 1.4 Honest summary for the OMM thesis

For the OMM thesis chapter, the right framing of this debate is:

> *"The empirical case for rough volatility (Gatheral-Jaisson-Rosenbaum 2018, Bayer-Friz-Gatheral 2016) is contested by recent work (Cont-Das 2022, Delemotte et al. 2024) showing that apparent roughness may stem from microstructure noise or estimator dependence. Rather than commit to either camp, we adopt a dynamics-agnostic methodology that handles both classical Markovian stochastic volatility (Heston, Bergomi, SABR) and rough/path-dependent volatility (rough Heston, rBergomi, rough Hawkes Heston) within a unified signature-based framework. The Koopman spectral roughness estimator of [own work, in progress] provides empirical guidance on the appropriate dynamics class for any given asset."*

This is a defensible position that doesn't lose to either camp.

---

## 2. Theoretical machinery for fractional / non-Markovian stochastic optimal control

There are roughly three lineages of mathematical machinery for handling fSDE control problems. Each has its own strengths and weaknesses.

### 2.1 Wick / white noise / Hida calculus (the oldest, theoretically deep, practically limited)

**Biagini, F., Hu, Y., Øksendal, B. & Zhang, T. (2008).** *Stochastic Calculus for Fractional Brownian Motion and Applications.* Springer Probability and its Applications. [Springer](https://link.springer.com/book/10.1007/978-1-84628-797-8).

**This is the canonical book on Wick / white noise / Hida calculus for fBm.** Develops:
- Stochastic integration with respect to fBm via Wick products (not pointwise products)
- Itô-type formulas for fBm
- The fractional Black-Scholes model and proof that it's arbitrage-free (using Wick integrals)
- Applications to finance, hydrology, biology

**Why Wick calculus matters historically**: it solves the arbitrage problem for fBm-driven asset price models. Pointwise integration with respect to fBm allows arbitrage; Wick integration does not.

**Why Wick calculus is unsatisfactory for finance**: the Wick integral doesn't correspond to a real trading strategy. As Bender et al. and others have pointed out, the "no arbitrage" obtained via Wick integration is mathematical artifact — you can't actually execute a Wick portfolio in a real market. So even though the math works out, it doesn't model anything tradeable.

**Hu, Y. & Øksendal, B. (2003).** *Fractional white noise calculus and applications to finance.* Infinite Dimensional Analysis, Quantum Probability and Related Topics 6(1), 1-32. The original applied paper using white noise / Hida calculus for fBm finance.

**Duncan, T. E., Hu, Y. & Pasik-Duncan, B. (2000).** *Stochastic calculus for fractional Brownian motion. I. Theory.* SIAM Journal on Control and Optimization 38(2), 582-612. [SIAM](https://epubs.siam.org/doi/10.1137/S036301299834171X). The theoretical foundation for the H > 1/2 case.

**Status for our purposes**: **Cite as historical/theoretical reference, do not use as the implementation framework.** The Wick calculus is restricted to H > 1/2 in most treatments, which is the wrong regime for rough volatility (we want H < 1/2). The white noise machinery is also computationally heavy and not amenable to learning algorithms.

### 2.2 Stochastic maximum principle for fBm (BSDE machinery, partially observed extensions)

**Hu, Y. & Peng, S. (2009).** *A stochastic maximum principle for processes driven by fractional Brownian motion.* [Stochastic Processes and their Applications 119(11), 3776-3796](https://www.researchgate.net/publication/227423326_A_stochastic_maximum_principle_for_processes_driven_by_fractional_Brownian_motion).

The original stochastic maximum principle for fBm. Uses Malliavin calculus on the fBm Cameron-Martin space to derive necessary conditions for optimal controls. Restricted to H > 1/2.

**Buckdahn, R. & Ma, J. (various).** Several papers extending the maximum principle to include jumps and partial observations.

**Han, B., Wong, H. Y. et al. (2019).** *Stochastic linear quadratic optimal control problem for systems driven by fractional Brownian motions.* [Optimal Control Applications and Methods, Wiley](https://onlinelibrary.wiley.com/doi/10.1002/oca.2523).

LQ control problem with fBm driving noise, derives a fractional Riccati equation. **This is one of the cleaner mathematical results** for fBm control, gives an explicit closed-form solution under linear-quadratic structure.

**Wang, Y., Zhang, Z. et al. (2024).** *The Global Maximum Principle for Optimal Control of Partially Observed Stochastic Systems Driven by Fractional Brownian Motion.* [SIAM Journal on Control and Optimization, vol 62](https://epubs.siam.org/doi/10.1137/22M1543203).

**Highly relevant.** Treats the partially observed case (POMDP) where the dynamics are driven by fBm. Derives the adjoint BSDE and necessary conditions for optimal partially observed control. **This is exactly the framework we want for OMM under rough vol with partially observed variance.**

**Sun, Y. & Wang, Z. (2025).** *A stochastic maximum principle for general controlled systems driven by mixed Brownian motions.* [Mathematical Control & Related Fields](https://www.aimsciences.org//article/doi/10.3934/mcrf.2025034). Mixed standard + fractional Brownian noise with H > 1/2.

**Sun, Y. (2023).** *Maximum principle for mean-field controlled systems driven by a fractional Brownian motion.* [OCAM Wiley](https://onlinelibrary.wiley.com/doi/10.1002/oca.3039). Mean-field extension.

**Tang, S. (2024).** *A Modified Maximum Principle for Control Systems Driven by Mixed Fractional Brownian Motion.* [arXiv:2312.11893](https://arxiv.org/html/2312.11893).

**Various (2020).** *Malliavin calculus used to derive a stochastic maximum principle for system driven by fractional Brownian and standard Wiener motions.* [De Gruyter Brill](https://www.degruyterbrill.com/document/doi/10.1515/rose-2020-2047/html?lang=en).

**Status for our purposes**: **Theoretically the right framework for rigorous treatment, but practically heavy.** The maximum principle gives necessary conditions via an adjoint BSDE; solving the BSDE in high dimensions is itself a hard problem. **Cite as the rigorous theoretical foundation, but do not implement directly.** Use signature methods (Section 3) as the practical alternative that approximates what the maximum principle would give.

### 2.3 Functional Itô calculus / path-dependent HJB (the rigorous treatment of path-dependent control)

**Dupire, B. (2009).** *Functional Itô Calculus.* Bloomberg working paper. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1435551). Republished as Dupire (2019), [Quantitative Finance 19(5), 721-729](https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1575974).

**The foundational paper.** Dupire extends Itô calculus to functionals of the path (not just the current value), introducing partial derivatives `∂_t F` (vertical, in time) and `∂_x F` (horizontal, in path perturbation), and proving a functional Itô formula. This is the rigorous calculus for path-dependent observables.

**Cont, R. & Fournié, D.-A. (2010).** Various papers extending Dupire's framework. Notably *Functional Itô calculus and stochastic integral representation of martingales* (Annals of Probability 41(1), 109-133, 2013) and *A functional extension of the Itô formula* (CRAS, 2010).

**Cont, R. (2017).** *Functional Itô calculus, path-dependence and the computation of Greeks.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304414917300881). [arXiv:1311.3881](https://arxiv.org/abs/1311.3881). Extends to Greeks computation for path-dependent options. **Useful for vega-equivalent computations under non-Markovian dynamics** — exactly what we need for the OMM inventory variance estimator under rough vol.

**Ekren, I., Touzi, N. & Zhang, J. (2014, 2016).** *Viscosity solutions of fully nonlinear parabolic path dependent PDEs.* [Annals of Probability]. Two papers (Part I and Part II). **The rigorous viscosity solution theory for the path-dependent HJB.** This is the path-dependent analog of the Crandall-Lions viscosity solution theory for the Markov HJB.

**Ekren, I., Keller, C., Touzi, N. & Zhang, J. (2014).** *Stochastic Control and Differential Games with Path-Dependent Reward Functionals.* [arXiv:1611.00589](https://arxiv.org/pdf/1611.00589). Extends path-dependent HJB to control with path-dependent rewards. **This is the right framework for OMM under non-Markov dynamics**: spread capture, inventory variance, and hedge cost can all be path-dependent functionals.

**Status for our purposes**: **Rigorous, applicable to fSDEs and any non-Markov dynamics, dynamics-agnostic in formulation.** Functional Itô calculus is the right *theoretical* framework — it generalizes to fSDEs, jump-diffusions, regime-switching, etc., without losing the HJB structure. The downside is that solving the path-dependent HJB numerically is hard. **In practice, signature methods are how we discretize functional calculus** — signatures are essentially a finite-dimensional projection of the path-functional space.

### 2.4 Direct fractional HJB approaches

**Bäuerle, N. & Desmettre, S. (2020).** *Portfolio Optimization in Fractional and Rough Heston Models.* [SIAM Journal on Financial Mathematics 11(1), 240-273](https://epubs.siam.org/doi/abs/10.1137/18M1217243). Also [Semantic Scholar](https://www.semanticscholar.org/paper/Portfolio-Optimization-in-Fractional-and-Rough-B%C3%A4uerle-Desmettre/3383bcd660c02a50b4a97434170a97abfe810630).

**Highly relevant.** Treats Merton's portfolio problem under (a) the fractional Heston model and (b) the rough Heston model, for power (CRRA) utility. Uses a representation of the fractional part followed by an approximation that casts the problem in classical stochastic control framework. Introduces a new model for the rough path scenario based on the **Marchaud fractional derivative**. **This is the closest existing reference to what we want to do for OMM.**

**Han, B., Wong, H. Y. & Xu, Z. Q. (2020).** *Merton's portfolio problem under Volterra Heston model.* [Finance Research Letters 39, 101631](https://www.sciencedirect.com/science/article/abs/pii/S1544612319312917). [arXiv:1909.02972](https://arxiv.org/abs/1909.02972).

Volterra Heston model — a generalization of rough Heston where the kernel is a general Volterra kernel. Merton portfolio problem solved via auxiliary stochastic process and martingale optimality principle. **The key technical move**: the value function is non-Markovian, but you can represent it via a finite-dimensional approximation of the variance kernel.

**Wang, Z., Wei, J. & Zhao, K. (2022).** *Mean-Variance Portfolio Selection Under Volterra Heston Model.* [Applied Mathematics & Optimization 85(2)](https://link.springer.com/article/10.1007/s00245-020-09658-3). Mean-variance version of the same problem.

**Various recent (2023-2025).** *Hamilton-Jacobi-Bellman equation for stochastic optimal control with random coefficients.* [Advances in Continuous and Discrete Models](https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-021-03674-5). General framework for stochastic HJB with random coefficients.

**Various (2024).** *Space-time fractional dynamics in Hamilton-Jacobi-Bellman model: a computational study.* [Springer](https://link.springer.com/article/10.1007/s12190-025-02654-2). Space-time fractional HJB for portfolio optimization.

**Note**: The "fractional HJB" terminology is sometimes used for the **Caputo / Riemann-Liouville fractional derivative** in time, which is a different generalization than fBm-driven dynamics. The Caputo-time-fractional HJB is mostly used in physics/anomalous diffusion contexts and is less directly relevant for finance.

**Status for our purposes**: **The Bäuerle-Desmettre and Han-Wong-Xu papers are the closest existing references to what we want to do.** They treat portfolio (not OMM) under rough Heston, but the techniques transfer. For an OMM analog of these results, we would need to combine:
1. The Bergault-Guéant (2019) OMM HJB structure
2. The Bäuerle-Desmettre / Han-Wong rough Heston techniques (Marchaud derivative, finite-dim approximation of the kernel)
3. Our SDRE solver for the local quadratic problem

**No existing paper does this combination**, which is good news for novelty.

### 2.5 The "stochastic HJB" with random coefficients (a useful intermediate)

**Peng, S. (1992).** *Stochastic Hamilton-Jacobi-Bellman Equations.* [SIAM J. Control Opt. 30(2), 284-304](https://epubs.siam.org/doi/10.1137/0330018). Foundational paper on backward stochastic HJB.

**Various (2013).** *A stochastic HJB equation for optimal control of forward-backward SDEs.* [arXiv:1312.1472](https://arxiv.org/pdf/1312.1472).

**Status**: Stochastic HJB with random coefficients is a useful intermediate framework — it doesn't require Markov property in the standard sense, but it doesn't fully embrace the path-dependent functional Itô machinery either. Useful as a stepping stone in the literature review.

---

## 3. Practical learning algorithms for high-dimensional fractional control

This is where the **signature-based methods** dominate and where our methodology should live. The key insight: **signatures convert path-dependent (non-Markov) dynamics into a finite-dimensional Markov representation**, allowing standard Koopman / SDRE / RL methods to apply.

### 3.1 Signature volatility models (the cleanest model class)

**Cuchiero, C., Gazzani, G. & Möller, J. (2024).** *Signature Volatility Models: Pricing and Hedging with Fourier.* [SIAM Journal on Financial Mathematics, vol 16](https://epubs.siam.org/doi/10.1137/24M1636952). [arXiv:2402.01820](https://arxiv.org/abs/2402.01820). [HAL](https://hal.science/hal-04435238v2/file/sigvol.pdf).

**Probably the single most relevant paper for our methodology.** They model volatility as a possibly infinite linear combination of time-extended signature elements of a Brownian motion. The framework includes:
- Stein-Stein, Bergomi, Heston (classical SV models) as special cases
- Path-dependent variants
- Non-Markovian / rough variants

They derive the **joint characteristic functional of log-price and integrated variance**, which enables **fast and accurate Fourier pricing and hedging beyond standard affine classes**, covering both Markovian and non-Markovian models.

**Why this matters for OMM**: signature volatility models give us a *unified* framework that includes Heston (where Bergault-Guéant's closed form applies) AND rough/path-dependent models (where it doesn't). **The same methodology works in both regimes.** This is essentially what we want.

### 3.2 Deep signatures for rough vol pricing/hedging

**Bayer, C., Pelizzari, L. et al. (2025).** *Pricing American options under rough volatility using deep-signatures and signature-kernels.* [arXiv:2501.06758](https://arxiv.org/html/2501.06758).

Deep-signature and signature-kernel learning methodologies for American option pricing under rough vol. **The combination of deep signatures + signature kernels is the current state of the art** for rough vol problems where you need both path-dependent state representation and a learning algorithm.

**Bonesini, O., Jacquier, A. & Pelizzari, L. (2025).** *Hedging with memory: shallow and deep learning with signatures.* [arXiv:2508.02759](https://arxiv.org/html/2508.02759v1). Direct head-to-head comparison of signature-based methods (shallow and deep) for hedging exotic derivatives under non-Markovian stochastic volatility.

**Key finding from this paper**: signatures as features in feedforward neural networks **outperform LSTMs in most cases with significantly less training compute**. Shallow learning approaches comparing direct hedging strategy learning from expected price signatures versus signature volatility models calibrated on volatility signatures find the **latter yields more accurate and stable results**. This is exactly the calibrated-signature-volatility-model approach we'd take.

**Various (2024).** *Rough Kernel Hedging.* [arXiv:2501.09683](https://arxiv.org/html/2501.09683).

> *"Models market dynamics as general geometric rough paths, yielding a fully model-free approach. By means of a representer theorem, theoretical guarantees on the existence and uniqueness of a global minimum of the resulting optimization problem are provided, with an analytic solution under highly general loss functions."*

**This is essentially what we want for OMM**: model-free, signature-kernel-based, with rigorous theoretical guarantees, scalable to high dimensions. **Cite as the methodological precedent.**

### 3.3 Deep hedging under rough vol (the deep RL competitor)

**Horvath, B., Teichmann, J. & Žurič, Ž. (2021).** *Deep Hedging under Rough Volatility.* [Risks 9(7), 138](https://www.researchgate.net/publication/349025494_Deep_Hedging_under_Rough_Volatility). The original "deep hedging meets rough vol" paper. Demonstrates that deep RL can solve hedging problems under rough vol where closed forms don't apply, but at significant compute cost.

**Various (2023).** *From Stochastic to Rough Volatility: A New Deep Learning Perspective on Hedging.* [Fractal and Fractional 7(3), 225](https://www.mdpi.com/2504-3110/7/3/225). Recent improvement: GRU-NN architecture specifically for non-Markovian hedging. Outperforms vanilla deep learning techniques.

**Buehler, H., Murray, P. & Wood, B. (2024).** *Deep Bellman Hedging.* [arXiv:2207.00932](https://arxiv.org/abs/2207.00932). Already in our reference stack as the Stage 6 comparison target.

**Various (2023).** *SigFormer: Signature Transformers for Deep Hedging.* [ACM ICAIF Proceedings](https://dl.acm.org/doi/10.1145/3604237.3626841). **Signature-Transformer hybrid** — uses signature features inside transformer architectures for hedging. State of the art for compute-vs-performance.

**Status for the Stage 6 Buehler comparison**: these are the deep RL competitors we'd benchmark against. The pitch is "signature-SDRE matches or outperforms these at fraction of compute, with interpretable controllers."

### 3.4 Signature-based volatility forecasting and calibration

**Various (2024).** *Forecasting volatility with machine learning and rough volatility: example from the crypto-winter.* [Digital Finance 6, 305-326](https://link.springer.com/article/10.1007/s42521-024-00108-1). Empirical validation of signature-based vol forecasting on crypto data.

**Horvath, B., Muguruza, A. & Tomas, M. (2021).** *Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models.* [Quantitative Finance 21(1)](https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1817974). The canonical "deep calibration of rough vol models" paper.

**Various (2025).** *Volatility Modeling with Rough Paths: A Signature-Based Alternative to Classical Expansions.* [arXiv:2507.23392](https://arxiv.org/html/2507.23392v2). Recent unification of signature-based vol modeling.

**Various (2023).** *Random neural networks for rough volatility.* [arXiv:2305.01035](https://arxiv.org/html/2305.01035). Random feature variant — much faster training, similar accuracy.

### 3.5 Evaluation of practical pragmatism

| Method | Theoretical rigor | High-dim scalability | Compute cost | Interpretability | Status for our use |
|---|---|---|---|---|---|
| Wick / Hida calculus | Very high (math) | Poor | High | Low | **Reject**: H>1/2 only, not implementable |
| Stochastic maximum principle (BSDE) | High | Poor | High | Medium | **Cite, don't implement**: theoretical foundation |
| Functional Itô calculus | High | Medium (via signatures) | Medium | Medium | **Cite as theoretical framework** |
| Direct rough HJB (Marchaud, Volterra) | High | Poor | Very high | Low | **Cite as closest existing references** |
| **Signature volatility models (Cuchiero et al.)** | High | Good | Low | High | **USE — primary methodological framework** |
| **Signature kernels + Koopman** | Medium-high | Excellent | Low | High | **USE — implementation engine** |
| Deep signature NNs | Medium | Excellent | Medium-high | Low | **USE for Stage 6 benchmark** |
| Deep RL (Buehler, etc.) | Low (heuristic) | Excellent | Very high | Very low | **Stage 6 comparison target only** |

**Verdict**: The pragmatic path is **signature-based Koopman-SDRE**, with signature volatility models as the theoretical framing (Cuchiero et al. 2024) and signature kernels / SDRE as the implementation engine. This is exactly what the existing repo machinery (`signature_features.py`, `streaming_sig_kkf.py`, `kronic_pomdp/`, the `fsde-identifiability` sibling) is designed to do.

---

## 4. Connections to existing sibling repos

The user has two sibling repos in the parent folder that are directly relevant. Both should be cited and integrated.

### 4.1 `fsde-identifiability` — nonparametric (μ, σ, H) estimation

**Path**: `/home/ed/SynologyDrive/Documents/Research/PE_Research/fsde-identifiability/`
**Status**: JMLR-target paper outline (Feb 2026)
**Key files**:
- `README.md` — overview
- `papers/jmlr_fsde_identifiability_outline.md` — paper outline
- `docs/spectral_roughness_theory.md` — Paper 6a, Koopman spectral roughness estimator (the rough vol debate arbiter)
- `docs/generator_estimator_walkthrough.md` — implementation walkthrough
- `src/sskf/nystrom_koopman.py` — Nyström eigenfunction extraction
- `src/sskf/tensor_features.py` — signature features
- `src/generator_estimator.py` — core estimator (Profile REML)
- `src/rough_paths_generator.py` — fBM simulation
- `src/hurst_estimators.py` — H estimation methods
- `src/spectral_hurst.py` — spectral H estimation

**Identifiability theorem (from the JMLR outline)**: For an fSDE `dX = μ(X)dt + σ(X)dB^H` with `H ∈ (0,1)`, `μ` continuous, `σ > 0` continuous, the triple `(μ, σ, H)` is **uniquely determined** by the law of the path. This extends Stroock-Varadhan to the non-Markov setting.

**Reported error rates** from the README:
- **H** (Hurst exponent) via aggregated variance: **2-3% error**
- **σ(x)** (diffusion) via local quadratic variation: **<3% error**
- **μ(x)** (drift) via fGN-whitened regression with eigenvalue constraint: **25-40% MRE** (much harder due to low SNR)

**Critical methodological insight (from the README)**: At typical `dt` and `H < 0.5`, **noise dominates signal** — drift contribution is `O(dt)` while noise contribution is `O(dt^H) >> O(dt)`. **Without fGN whitening, regression learns the noise, not the drift.** The whitening procedure:
1. Estimate `H` from trajectory
2. Build fGN correlation matrix
3. Cholesky factorize: `Σ = L · L^T`
4. Whiten: `dx_w = L^{-1} · dx`
5. Regress whitened features on whitened `dx`

**Critical for our OMM use**: when we eventually deploy on rough/observable price data, **the inventory variance estimator must use whitened estimates**, not naive realized variance. The naive estimator is biased at H ≠ 0.5.

**For our OMM work, this repo provides**:
- The Koopman generator estimation machinery (already implemented)
- The signature feature representation
- The Hurst estimator
- The proven identifiability of (μ, σ, H) from observed paths
- The whitening discipline that avoids noise-driven bias

**Action items for our OMM work**:
1. **Stage 5+**: import `fsde-identifiability/src/sskf/tensor_features.py` and `nystrom_koopman.py` as the signature feature backend
2. **Stage 5+**: use the Hurst estimator from `fsde-identifiability/src/hurst_estimators.py` to characterize the dynamics regime of the env
3. **Citation for the OMM thesis**: Mehrez, E. (in prep, 2026), "Nonparametric Identification of Fractional Stochastic Differential Equations via Signature Methods" — the JMLR-target paper

### 4.2 `fSDE_video_gen` — fSDE generative models for time series

**Path**: `/home/ed/SynologyDrive/Documents/Research/PE_Research/fSDE_video_gen/`
**Status**: Paper 1 (signature distance metric for video) and Paper 2 (fSDE time series generative models) in progress
**Key files for our use**:
- `docs/paper2_fsde_timeseries.md` — Paper 2 outline (most relevant)
- `docs/paper1_signature_metric.md` and `docs/paper1_signature_metric_theory.md` — Paper 1 (signature distance metric)
- `NEXT_STEPS.md` — current state and priorities
- `video_diffusion/SIGNATURE_ATTENTION_RESEARCH.md` — signature attention research

**Paper 2's relevance to our OMM work** (from `paper2_fsde_timeseries.md`):

> "Replace `W_t` with `B^H_t` (fractional Brownian motion): `dx_t = f(x_t, t) dt + g(t) dB^H_t`. Now the noise process itself encodes temporal structure. **H > 0.5 (persistent)**: Increments are positively correlated → trending behavior (momentum in finance, slow physiological rhythms). **H < 0.5 (anti-persistent)**: Increments are negatively correlated → mean-reverting behavior (volatility, bounded oscillations). **H = 0.5**: Standard BM (recovered as special case)."

The Paper 2 outline includes:
- **Fractional score matching** (correct objective for fBm-driven forward processes)
- **Reverse-time fSDE** (the non-Markovian reverse process — explicitly noted as "the key theoretical contribution. If we can derive and implement this correctly, it's a genuine novelty.")
- **Signature matching guarantee** (signatures verify the generated paths match the data distribution)

The **reverse-time fSDE** is technically related to our problem: it's the path-dependent backward SDE that arises when you try to invert a fractional forward process. Anderson's classical reverse-time SDE assumes Markov property; for fBm, the reverse process is non-Markovian and has a Volterra-type kernel:

```
dx_t = [f(x_t, t) − g²(t)(s_θ(x_t, t) + ∫_0^t K_H(t,s) dx_s)] dt + g(t) d B̄^H_t
```

For our OMM work this isn't directly used, but **the technical machinery for handling the non-Markov reverse dynamics is the same machinery we'd need for path-dependent OMM HJBs**. The two papers (Paper 2 fSDE generative + OMM SDRE) share infrastructure.

**Action items**:
1. **Stage 5+**: cross-check with the fSDE generative work to see if any of the score-matching / reverse-time machinery is reusable
2. **Citation**: Mehrez, E. & Rozwood, P. (in prep, 2026), "Fractional SDE Generative Models for Time Series" — Paper 2

### 4.3 The Koopman spectral roughness paper (Paper 6a)

**Location**: `fsde-identifiability/docs/spectral_roughness_theory.md`
**Status**: In-progress theory document, February 2026

**Key result claimed**: For an fBm with Hurst parameter `H ∈ (0,1)`, the eigenvalues `{λ_k}` of the Koopman generator satisfy
```
|λ_k| ~ k^{-(2H+1)} as k → ∞
```
This enables direct extraction of `H` from observed eigenvalue decay, **independent of the Cont-Das microstructure noise contamination**, because microstructure noise affects only the high-frequency tail of the spectrum (high `k`) while `H` is determined by the asymptotic decay rate that can be read from the low/middle eigenvalues.

**Why this matters for OMM**: this gives us a **principled way to estimate the dynamics class** of any observed price series before deploying the OMM controller. The procedure:
1. Observe a price path
2. Compute the Koopman generator via KGEDMD on signature features
3. Extract eigenvalue decay rate
4. Read off `H`
5. If `H ≈ 0.5`: classical Markovian dynamics, use BG-style estimator
6. If `H < 0.5`: rough vol, use signature-based estimator
7. If `H > 0.5`: long-memory but smooth, use a smoothed estimator

**Action items**:
1. **Stage 5+**: integrate the spectral roughness estimator into the OMM pipeline as a "regime detector"
2. **Citation**: Mehrez, E. (in prep, 2026), "Spectral Estimation of Path Roughness via Koopman Operators" — Paper 6a

---

## 5. Why fSDE-only is still a POMDP

This is a question the user explicitly asked, so let me answer it clearly.

The short answer: **fSDE control is a POMDP in the strictest formal sense, even when the agent observes the price path directly**, because the "true state" of an fSDE is not the current price but the **infinite history of fractional noise increments**. The agent can never directly observe this hidden state — only its projection onto the observable price.

The longer answer involves three considerations:

### 5.1 fSDEs typically still have hidden state beyond fBm

Most fSDE models used in finance — rough Heston, rough Bergomi, rough Hawkes Heston — have **both** fractional driving noise AND additional hidden components like the variance process itself. For these models:

| Model | Hidden? | Markov in raw state? | POMDP? |
|---|---|---|---|
| Heston | Variance `ν_t` is hidden | Yes, in `(S, ν)` | Yes (classical) |
| Rough Heston | Variance `ν_t` is hidden AND non-Markov | No (variance has memory) | Yes (non-Markov) |
| Rough Bergomi | Forward variance curve is hidden AND non-Markov | No | Yes (non-Markov) |
| Pure fBm log-spot model | Nothing hidden, but non-Markov | No (memory in increments) | "Path-observable POMDP" |
| Local volatility | Nothing hidden | Yes (in `(S, t)`) | No (it's an MDP) |

**Even for rough vol models without explicit additional hidden state**, the past noise increments are not directly observable — only the price is. The agent can compute the price's path-dependent statistics (signatures, realized variance, etc.) but cannot recover the underlying fBm increments, because multiple noise paths give the same observed price.

### 5.2 The lifted-state perspective makes fSDE look like an MDP — in path space

There's a sense in which an fSDE "is" an MDP if you take the path itself as the state. The full path `{X_s : 0 ≤ s ≤ t}` IS a Markov state for the future dynamics. This is the **functional Itô calculus** perspective (Dupire 2009): the path is the state, and the dynamics are Markovian *in path space*.

But this is not finite-dimensional, and the agent cannot store the whole path. Practically, the agent must use a **finite-dimensional projection** of the path, e.g., signature features. The projection is *lossy*: from `Sig(path)` you cannot recover all the path information, only the relevant low-order moments. So:

- In *infinite-dimensional path space*: fSDE is an MDP, fully observable.
- In *finite-dimensional signature space*: fSDE is a *POMDP*, where the "hidden state" is the high-order signature components that the truncation discards.
- In *raw current-state space* `(S_t)`: fSDE is a POMDP with infinite-dimensional hidden state (the entire history).

**The three views are equivalent in theory but very different in practice.** The agent operates on the finite signature representation, and this is a POMDP because the truncation is lossy.

### 5.3 The Koopman formulation embraces the POMDP framing naturally

The whole point of the Koopman methodology is that **it works on observable functionals of the state** — you don't need to know the underlying state, you just need to compute observables and learn how they evolve. The Koopman generator `L` on observables `f: S → ℝ` is well-defined whether `S` is the raw state space or the lifted (signature) space, and the methodology is the same:

- For Markov diffusions: `L f(x) = μ(x) f'(x) + (1/2) σ²(x) f''(x)` (the standard infinitesimal generator)
- For fSDEs in lifted space: `L̃ f(s) = lim_{h→0} (E[f(s_{t+h}) | s_t = s] − f(s)) / h` where `s_t = Sig(path up to t)`. The generator is well-defined on signature observables even though the underlying process is non-Markov.

**The Koopman / signature combination handles fSDEs by lifting to a Markov representation in the signature space, then learning the generator from data in that space.** This is what `fsde-identifiability` does, what `signature_features.py` implements, and what `kronic_pomdp/` is built on. We don't need to write down a path-dependent HJB explicitly; the Koopman methodology gives us local moments without needing the closed-form generator.

### 5.4 The bottom line for the OMM work

**Yes, fSDE-driven OMM is still a POMDP** — and that's exactly why we want it. The POMDP framing justifies the entire approach:

1. **Koopman on observables**: works on any continuous-time stochastic process, Markov or not, hidden state or not.
2. **Signature features**: the natural finite-dimensional state representation that handles path-dependence.
3. **No filtering needed for the hidden noise**: signatures absorb the path information without explicit filtering.
4. **Estimation from data**: KGEDMD learns the generator on signatures from observed (state, action, transition) tuples.
5. **The CdC identity** (`σ²(x) = L(x²) − 2x L(x)`) gives us local moments — including the inventory variance — directly from the learned generator, **without needing to know whether the dynamics are Heston, rough Heston, or anything else.**

For the OMM thesis chapter, the framing becomes:

> *"We treat the options market making problem as a POMDP under continuous-time stochastic dynamics, where the underlying volatility process may be Markovian (e.g., Heston) or path-dependent / fractional (e.g., rough Heston, rough Bergomi, fBm log-spot). The agent observes the price path and computes signature features as the state representation. The Koopman generator on signature observables is learned from data via KGEDMD; the local inventory variance for the SDRE controller is computed from the Carré du Champ identity applied to the learned generator. This framework recovers the Bergault-Guéant (2019) closed-form optimum on Heston dynamics (where it applies) and extends naturally to rough/path-dependent dynamics where no published closed form is available."*

That's the strongest possible framing.

---

## 6. Recommendation for our OMM SDRE methodology

Given all of the above, here's my recommendation for how the OMM SDRE methodology should treat dynamics generality:

### 6.1 The pragmatic stack

1. **Theoretical framework**: signature volatility models (Cuchiero-Gazzani-Möller 2024). This is the cleanest unified framework that covers Heston, Bergomi, rough Heston, rough Bergomi, etc.

2. **Numerical engine**: Koopman generator estimation on signature features (the existing repo's machinery, the `fsde-identifiability` sibling, KGEDMD from Klus et al. 2020).

3. **Inventory variance estimator**: Carré du Champ identity applied to the learned Koopman generator on signature observables. **Dynamics-agnostic by construction**.

4. **Optimal quote computation**: SDRE local Riccati at each state, with the inventory variance from step 3 plugged in. **The same closed-form formula `δ⁻* = 1/k + (γ_local/2) σ²_inv (T−t) q` applies across all dynamics classes.** Only the `σ²_inv` estimator changes.

5. **General utility extension**: Davis-Lleo (2014) local Arrow-Pratt construction. **Orthogonal to the dynamics generality.**

### 6.2 What to NOT pursue

- **Wick / Hida calculus**: theoretical dead end for our purposes (H > 1/2 only, not implementable for learning).
- **Fractional HJB via Caputo derivatives**: wrong generalization for finance.
- **Direct path-dependent HJB solver**: too expensive in high dimensions; signatures are the practical alternative.
- **BSDE / stochastic maximum principle implementation**: cite as theoretical foundation, but don't implement directly.

### 6.3 Stage scoping

- **Stage 4 v2** (next): Heston only, with the controller written abstractly enough to support any inventory variance estimator. Use **Heston-specific** and **empirical sliding-window** estimators. Validate against Bergault-Guéant closed form. Confirm the abstract interface works.
- **Stage 5**: Rough Heston / rough Bergomi env, with Koopman-CdC inventory variance estimator on signature features. **The methodology demonstration**: same controller works on dynamics where Bergault-Guéant doesn't apply.
- **Stage 6**: Buehler deep RL comparison on both Heston and rough vol envs. Cost vs performance.
- **Stage 7**: Real Alpaca data. The methodology handles this without re-derivation.

### 6.4 The publication strategy

The right journal for the OMM work, given this framing, is probably:

- **Quantitative Finance** (the canonical Bergault-Guéant venue) — if we frame as "extending Bergault-Guéant to general dynamics"
- **Mathematical Finance** — if we frame as "rigorous treatment of OMM under path-dependent dynamics"
- **SIAM Journal on Financial Mathematics** — if we frame as "numerical methods + computational comparison vs deep RL"
- **Finance and Stochastics** — if we frame more theoretically

The Buehler audience (JP Morgan) cares about practical performance + interpretability, so the **Quantitative Finance** route is probably the right primary target, with a longer / more theoretical version optionally for SIAM JFM.

---

## 7. Open questions (for future research, not blocking Stage 4 v2)

1. **Does the Koopman generator on signature features converge to the "true" generator of the underlying SDE/fSDE in the appropriate limit?** This is a consistency question. The user's `fsde-identifiability` paper has results in this direction; need to check if they cover the OMM-relevant case.

2. **What's the right truncation order for the signature features?** Higher order = more information but more parameters to estimate. There's an optimal trade-off depending on the data size and the smoothness of the value function.

3. **How does the Cont-Das microstructure noise critique apply to our Koopman-CdC inventory variance estimator?** Specifically: if the apparent roughness is artifact, does our estimator over-penalize inventory by treating fake roughness as real volatility? **Worth a Stage 5 robustness experiment.**

4. **Can we extend the Bergault-Guéant single-asset framework to multi-strike under signature features?** The natural extension uses portfolio vega `𝒱^π = Σ q^i 𝒱^i` as the relevant inventory state, with the Koopman-CdC estimator giving the local variance of `𝒱^π`. Same controller, different state. **Worth a Stage 5 or Stage 6 experiment.**

5. **How does the local Arrow-Pratt construction interact with non-Markov dynamics?** The Davis-Lleo argument is for Markov diffusions; the path-dependent generalization needs care. **Worth a literature search.**

---

## 8. Action items for the OMM derivation note revision

Based on this synthesis, the `docs/derivation_omm_sdre_v2.md` revision should:

1. **Replace "Heston" with "general continuous-time stochastic process (Markov or not, hidden state or not)" wherever the math allows.**
2. **State the inventory penalty in terms of `σ²_inv(s_t)` as an abstract local-variance estimator.**
3. **Add a Section 7.5 enumerating estimator options**: Heston-specific (BG), empirical sliding-window, Koopman-CdC on signatures.
4. **Add a Section 11 on the POMDP framing** answering "why is this still a POMDP for fSDEs?" using the material in Section 5 above.
5. **Add a Section 12 on dynamics regime detection** pointing to the Koopman spectral roughness estimator from `fsde-identifiability/docs/spectral_roughness_theory.md`.
6. **Update the audit checklist** to include items about the dynamics-agnostic estimator interface.
7. **Update the implementation map** with the `inventory_variance_estimator` callable interface.
8. **Cite this references doc** explicitly so codex / future readers can find the literature.

---

## 9. Changelog

| Date | Update |
|---|---|
| 2026-04-08 | Initial draft (Claude). Synthesizes web search + sibling repo content + the Bergault-Guéant references already in `docs/refs/`. |
