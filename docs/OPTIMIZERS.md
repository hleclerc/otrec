# Optimiseurs de reconstruction

Ce module fournit plusieurs algorithmes d'optimisation pour la reconstruction 2D par transport optimal.

## Algorithmes disponibles

### 1. Gradient Descent (baseline)
Descente de gradient simple avec pas fixe.

```python
from optimizers import GradientDescent

optimizer = GradientDescent(lr=0.2, nb_steps=500)
rec.diracs(optimizer=optimizer)          # ou rec.disks(...) pour le modèle disques
```

**Pros**: Simple, fiable, bon baseline  
**Cons**: Convergence lente, pas adaptatif  
**Bon pour**: Petits problèmes, baseline de comparaison

### 2. Gradient Descent + Line Search
Gradient descent avec recherche de pas par backtracking (Armijo).

```python
from optimizers import GradientDescentLineSearch

optimizer = GradientDescentLineSearch(lr=1.0, nb_steps=300, c1=1e-4, rho=0.5)
rec.diracs(optimizer=optimizer)          # ou rec.disks(...) pour le modèle disques
```

**Pros**: Converge plus vite que GD, pas d'hyperparamètres critiques  
**Cons**: Légèrement plus cher par itération (line search)  
**Bon pour**: Cas généraux, bon compromis vitesse/simplicité

### 3. Adam (Adaptative Moment Estimation)
Optimiseur adaptatif moderne avec moments exponentiels.

```python
from optimizers import Adam

optimizer = Adam(lr=5e-3, nb_steps=300, beta1=0.9, beta2=0.999)
rec.diracs(optimizer=optimizer)          # ou rec.disks(...) pour le modèle disques
```

**Pros**: Converge bien sur différentes topologies, robuste  
**Cons**: Plus d'hyperparamètres, peut être trop agressif  
**Bon pour**: Problèmes variés, quand on ne connaît pas bien la perte

### 4. L-BFGS (Recommended)
Limited-memory BFGS via scipy. Utilise des approximations de Hessien.

```python
from optimizers import LBFGS

optimizer = LBFGS(max_iter=200, ftol=1e-8)
rec.diracs(optimizer=optimizer)          # ou rec.disks(...) pour le modèle disques
```

**Pros**: Converge très rapidement (~5-10x plus d'itérations), excellente qualité  
**Cons**: Plus coûteux par itération (calcul du Hessien approché)  
**Bon pour**: Production, quand on veut la meilleure qualité rapidement

## Résultats de benchmark

Benchmark sur 10,000 diracs (100 angles × 100 bins):

```
Optimizer           | Steps | Time  | Final Loss | Speedup
--------------------|-------|-------|-----------|----------
Gradient Descent    | 500   | ~5s   | 0.0068    | 1.0x
GD + Line Search    | 180   | ~2s   | 0.0045    | 2.5x
Adam                | 200   | ~2.5s | 0.0050    | 2.0x
L-BFGS              | 34    | ~0.8s | 0.0027    | 6.2x ⭐
```

**Recommandation**: L-BFGS pour la plupart des cas (speedup 6x, meilleure qualité).

## Utilisation

### Utilisation simple

```python
from Reconstruction import Reconstruction
from Sinogram import Sinogram
from optimizers import LBFGS

sinogram = Sinogram(...)

rec = Reconstruction(sinogram, extent=1.0)
rec.random_points(50)
rec.diracs(optimizer=LBFGS())            # étape 1 : modèle diracs
print(rec.loss(), rec.summary())
```

Sans `optimizer`, chaque étape utilise le L-BFGS par défaut de l'objet, réglé à la construction
(`Reconstruction(..., max_iter=..., ftol=...)`) ou à l'appel (`rec.diracs(max_iter=300)`).

### Enchaîner les algorithmes

Chaque étape part du nuage laissé par la précédente, et renvoie `self` :

```python
rec = Reconstruction(sinogram, radius=0.15, record=True)
rec.random_points(60).diracs(max_iter=100).disks(max_iter=300)
rec.export_html("out.html")              # rayons fixes exportés après une étape disques
```

### Avec monitoring

```python
def my_callback(step, positions):
    print(f"Step {step}: loss = {rec.loss(points=positions):.8f}")

rec.diracs(optimizer=LBFGS(max_iter=200), callback=my_callback)
```

## Ajouter un nouvel optimiseur

Hériter de `Optimizer` et implémenter `minimize()`:

```python
from optimizers import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, ...):
        # vos hyperparamètres

    def minimize(self, scalar_loss, x0, callback=None):
        """
        Args:
            scalar_loss: function(x) -> float
            x0: array initial
            callback: optionnel function(step, x) appelé à chaque itération

        Returns:
            array optimisé
        """
        x = x0.copy()
        grad = driver.grad(scalar_loss)

        for step in range(nb_steps):
            g = grad(x)
            x = x - lr * g  # mise à jour
            if callback is not None:
                callback(step, x)

        return x
```

## Benchmarking

Pour lancer les benchmarks complets:

```bash
cd applications/reconstruction
python -c "from benchmark import *; benchmark_optimizers(nb_diracs=10000)"
```

Cela génère des PNG avec les courbes de convergence et l'analyse de scaling.

## Références

- [LBFGS - Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
- [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Backtracking Line Search](https://en.wikipedia.org/wiki/Backtracking_line_search)
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
