# Reconstruction optimizers

This module provides several optimization algorithms for 2D reconstruction via optimal transport.

## Available algorithms

### 1. Gradient Descent (baseline)
Plain gradient descent with a fixed step size.

```python
from optimizers import GradientDescent

optimizer = GradientDescent(lr=0.2, nb_steps=500)
rec.diracs(optimizer=optimizer)          # or rec.disks(...) for the disks model
```

**Pros**: Simple, reliable, good baseline  
**Cons**: Slow convergence, non-adaptive step  
**Good for**: Small problems, comparison baseline

### 2. Gradient Descent + Line Search
Gradient descent with backtracking (Armijo) line search.

```python
from optimizers import GradientDescentLineSearch

optimizer = GradientDescentLineSearch(lr=1.0, nb_steps=300, c1=1e-4, rho=0.5)
rec.diracs(optimizer=optimizer)          # or rec.disks(...) for the disks model
```

**Pros**: Converges faster than GD, no critical hyperparameters  
**Cons**: Slightly more expensive per iteration (line search)  
**Good for**: General cases, good speed/simplicity trade-off

### 3. Adam (Adaptative Moment Estimation)
Modern adaptive optimizer with exponential moments.

```python
from optimizers import Adam

optimizer = Adam(lr=5e-3, nb_steps=300, beta1=0.9, beta2=0.999)
rec.diracs(optimizer=optimizer)          # or rec.disks(...) for the disks model
```

**Pros**: Converges well on different topologies, robust  
**Cons**: More hyperparameters, can be too aggressive  
**Good for**: Varied problems, when the loss is not well known

### 4. L-BFGS (Recommended)
Limited-memory BFGS via scipy. Uses Hessian approximations.

```python
from optimizers import LBFGS

optimizer = LBFGS(max_iter=200, ftol=1e-8)
rec.diracs(optimizer=optimizer)          # or rec.disks(...) for the disks model
```

**Pros**: Converges very quickly (~5-10x fewer iterations), excellent quality  
**Cons**: More expensive per iteration (approximate Hessian computation)  
**Good for**: Production, when you want the best quality quickly

## Benchmark results

Benchmark on 10,000 diracs (100 angles × 100 bins):

```
Optimizer           | Steps | Time  | Final Loss | Speedup
--------------------|-------|-------|-----------|----------
Gradient Descent    | 500   | ~5s   | 0.0068    | 1.0x
GD + Line Search    | 180   | ~2s   | 0.0045    | 2.5x
Adam                | 200   | ~2.5s | 0.0050    | 2.0x
L-BFGS              | 34    | ~0.8s | 0.0027    | 6.2x ⭐
```

**Recommendation**: L-BFGS for most cases (6x speedup, best quality).

## Usage

### Simple usage

```python
from Reconstruction import Reconstruction
from Sinogram import Sinogram
from optimizers import LBFGS

sinogram = Sinogram(...)

rec = Reconstruction(sinogram, extent=1.0)
rec.random_points(50)
rec.diracs(optimizer=LBFGS())            # step 1: diracs model
print(rec.loss(), rec.summary())
```

Without `optimizer`, each step uses the object's default L-BFGS, configured at construction
(`Reconstruction(..., max_iter=..., ftol=...)`) or at call time (`rec.diracs(max_iter=300)`).

### Chaining algorithms

Each step starts from the point cloud left by the previous one, and returns `self`:

```python
rec = Reconstruction(sinogram, radius=0.15, record=True)
rec.random_points(60).diracs(max_iter=100).disks(max_iter=300)
rec.export_html("out.html")              # fixed radii exported after a disks step
```

### With monitoring

```python
def my_callback(step, positions):
    print(f"Step {step}: loss = {rec.loss(points=positions):.8f}")

rec.diracs(optimizer=LBFGS(max_iter=200), callback=my_callback)
```

## Adding a new optimizer

Subclass `Optimizer` and implement `minimize()`:

```python
from optimizers import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, ...):
        # your hyperparameters

    def minimize(self, scalar_loss, x0, callback=None):
        """
        Args:
            scalar_loss: function(x) -> float
            x0: initial array
            callback: optional function(step, x) called at each iteration

        Returns:
            optimized array
        """
        x = x0.copy()
        grad = driver.grad(scalar_loss)

        for step in range(nb_steps):
            g = grad(x)
            x = x - lr * g  # update
            if callback is not None:
                callback(step, x)

        return x
```

## Benchmarking

To run the full benchmarks:

```bash
cd applications/reconstruction
python -c "from benchmark import *; benchmark_optimizers(nb_diracs=10000)"
```

This generates PNGs with the convergence curves and the scaling analysis.

## References

- [LBFGS - Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
- [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Backtracking Line Search](https://en.wikipedia.org/wiki/Backtracking_line_search)
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
