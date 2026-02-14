# CartPole Stabilization - Systematic Testing Results

## Options Tested

### Option A: Periodic Kernel (Raw θ with periodic distance)

**Configuration:**

- State: [x, dx, θ, dθ] (dim 4, raw)
- Kernel: RBF with periodic distance for θ dimension
- Centers: 50
- Q: Sparse [0, 0, 10, 0]

**Result:** **FAILURE**

```
[RICCATI FAIL] LinAlgError: Failed to find a finite solution.
Final Theta: 57.9862 rad (spun wildly)
```

**Analysis:** Periodic kernel alone doesn't resolve the fundamental issue. The learned (A, B_eff) pair remains non-stabilizable.

---

### Option B: Lifted Coordinate Collection

**Configuration:**

- State: [x, dx, cos(θ), sin(θ), dθ] (dim 5, lifted)
- Data collection: `compute_lifted_derivative()` - collect transitions in lifted space
- Centers: 50 (reduced from 100 per user suggestion)
- Q: Sparse [0, 0, 10, 10, 0]

**Result:** **PARTIAL SUCCESS**

```
Learned Parameters:
- |A| = 6.96
- Max|λ(A)| = 1.064 (slightly unstable)
- |B_eff(target)| = 0.0824 (still low!)
- Spectral Norm: λ_max = 1.064

Control:
✅ NO RICCATI FAILURES! (ARE solver succeeded)
❌ Failed to stabilize
Final Theta: 1.8862 rad
```

**Analysis:** **Major improvement!**

1. ✅ ARE solver works (found finite solution)
2. ✅ System is controllable in lifted space
3. ❌ B matrix still too small (0.08 vs Heston's typical ~0.3-0.5)
4. ❌ Controller attempting control but insufficient authority

---

## Key Insights

### Why Option B is Better

1. **Correct Dynamics**: Collecting data in lifted coordinates ensures:

   ```
   d(cos θ)/dt = -sin(θ) · dθ/dt
   d(sin θ)/dt = cos(θ) · dθ/dt
   ```

   These are the actual dynamics that get learned.

2. **Numerical Stability**: Fewer centers (50) improved conditioning

3. **No ARE Failures**: This confirms the system IS stabilizable when modeled correctly

### Why B Matrix is Still Small

**Hypothesis:** The data collection near equilibrium (θ ≈ 0) has:

- cos(θ) ≈ 1 (barely changes)
- sin(θ) ≈ θ (small)

With sparse Q penalizing [cos, sin] and cos barely moving, the controller doesn't learn strong B authority.

---

## Proposed Next Steps

### Option B.1: Penalize Angular Velocity Instead

```python
# Current (doesn't work well)
Q_physical = np.diag([0.0, 0.0, 10.0, 10.0, 0.0])  # penalize cos/sin

# Proposed
Q_physical = np.diag([0.0, 0.0, 0.0, 0.0, 10.0])    # penalize dθ
```

**Rationale:** Near equilibrium, angular velocity is more sensitive to control than cos/sin values.

### Option B.2: Increase Data Exploration

Current: Only 500 samples for Stage 2 (local linearization)  
Proposed: 2000 samples with wider angle range (±0.2 rad)

### Option C: Simplify to 2D Problem First

Test with just [θ, dθ] - pole balancing without cart position.
If this works, gradually add cart dynamics.

### Option D: Use Continuous-Time Formulation

Set `ctrl.use_generator = True` to use CARE instead of DARE.
This worked better for Heston (continuous stochastic process).

---

## Recommendation

**Try Option B.1 first** (penalize angular velocity). It's a one-line change that directly addresses the low B authority issue. If that doesn't work, try B.2 (more data) before moving to the more complex options.
