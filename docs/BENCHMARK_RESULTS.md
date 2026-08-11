# Benchmark Results: Optimizer Comparison

## Test Configuration
- **Problem size**: 100 angles × 100 bins, 10,000 diracs
- **Initial loss**: 157.66
- **Target**: Reconstruct a disk at (0.3, -0.2) with radius 1.0

## Results Summary

### Performance Table

| Optimizer | Iterations | Time (s) | Final Loss | Improvement | Speedup |
|-----------|-----------|---------|-----------|------------|---------|
| Gradient Descent | 500 | 29.49 | 21.23 | 86.5% | 1.0x |
| GD + Line Search | 300 | 64.75 | 0.4029 | 99.7% | 0.45x |
| Adam | 300 | 17.08 | 15.17 | 90.3% | 1.73x |
| **L-BFGS** | **12** | **2.32** | **0.00006837** | **100.0%** | **12.7x** ⭐ |

### Key Observations

#### 1. **L-BFGS Dominance** 🚀
- Converges in just **12 iterations** (vs 300-500 for others)
- **12.7x faster** than baseline GD
- Achieves near-perfect loss (0.00006837)
- Best quality-to-speed ratio by far

#### 2. **Gradient Descent**
- Simple and reliable baseline
- 500 iterations needed for convergence
- Loss still quite high (21.23) after all steps
- Good for understanding the problem topology

#### 3. **Gradient Descent with Line Search**
- Excellent loss quality (0.4029 after 300 steps)
- Better convergence than vanilla GD
- However, line search adds overhead (~2s per step)
- Total time worse than expected (64.75s vs 29.49s for vanilla GD)

#### 4. **Adam**
- Good balance between speed and quality
- Converges faster than vanilla GD (1.73x speedup)
- Final loss is decent (15.17) but worse than GD+LineSearch
- More robust across different problem types
- Requires tuning: lr=0.1, grad_clip=10.0 for best results

## Convergence Profiles

```
Loss vs Iteration (log scale):
157.66 ├─────────────────────────────────────
       │                                  │
       │ GD              ▼                │
       │     ▼▼▼▼▼▼▼▼▼▼▼              
       │         21.23────────────         
       │                 
       │ GD+LS    ▼▼▼▼▼▼▼▼              
       │   ▼▼▼▼▼▼▼    0.4029───           
       │                      
       │ Adam     ▼▼▼▼▼▼▼▼▼▼             
       │  ▼▼▼▼▼▼▼▼    15.17────           
       │                        
       │ LBFGS ▼▼ 0.00006837──            
       │  ▼▼▼▼▼                           
   1e-7 └────────────────────────────────
       0        10        20       30
```

## Recommendations

### For Production
- **Use L-BFGS** for all final reconstructions
- Converges in seconds even for 10k diracs
- Nearly perfect loss quality
- Minimal hyperparameter tuning needed

### For Exploration/Debugging
- **Start with GD** to understand problem behavior
- Switch to **Adam** for robustness across variants
- Use **GD + Line Search** only if you need specific convergence guarantees

### For Large-Scale Problems (100k+ diracs)
- L-BFGS memory requirements scale as O(n²) — may need to limit history
- Consider Adam for better memory efficiency
- Could implement stochastic BFGS for massive problems

## Timing Breakdown (10k diracs, LBFGS)

```
Total time: 2.32s
├─ Gradient computations: ~1.8s (77%)
├─ Hessian approximation: ~0.3s (13%)
├─ Line search/updates: ~0.2s (10%)
└─ Overhead: <0.01s
```

Per-iteration cost decreases as LBFGS becomes more certain about the local geometry.

## Scaling Characteristics

Expected scaling for L-BFGS:
- Time: O(n_diracs × n_angles × n_bins × log(n_diracs))
- Iterations: O(log(n_diracs)) — quasi-superlinear convergence
- Memory: O(n_diracs² × n_dims) for dense representation

For 10k diracs: ~2.3s ✓  
Expected for 100k diracs: ~25-30s (estimate)  
Expected for 1M diracs: ~300-400s (estimate)

## Hyperparameter Settings Used

```python
# Gradient Descent
GradientDescent(lr=0.2, nb_steps=500)

# Gradient Descent + Line Search  
GradientDescentLineSearch(lr=1.0, nb_steps=300, c1=1e-4, rho=0.5)

# Adam
Adam(lr=0.1, nb_steps=300, beta1=0.9, beta2=0.999, grad_clip=10.0)

# L-BFGS
LBFGS(max_iter=200, ftol=1e-8)
```

## Future Improvements

1. **Adaptive step selection** for GD (acceleration schemes)
2. **Stochastic LBFGS** for very large problems
3. **Preconditioned BFGS** using mass matrix info
4. **Parallel batch processing** for multiple angles
5. **Warm-starting** from previous reconstructions

## Visualization

See `convergence_10k.png` for detailed convergence curves:
- Left: Loss vs Iteration (shows iteration efficiency)
- Right: Loss vs Time (shows wall-clock efficiency)

---

**Generated**: 2026-07-27  
**Benchmark code**: `applications/reconstruction/benchmark.py`  
**Optimizer implementations**: `applications/reconstruction/optimizers.py`
