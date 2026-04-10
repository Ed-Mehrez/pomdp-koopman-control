"""BBG-faithful option market-making benchmark.

Implements Baldacci, Bergault & Gueant (2020) with:
- multi-option book (N strikes x M maturities)
- logistic intensity functions
- portfolio-vega state reduction
- 3D HJB solver on (t, nu, V^pi) grid
"""
