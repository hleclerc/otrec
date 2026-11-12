# Reconstruction rapide (otrec) : round 4

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour plus de détails.
> Écris n'importe quoi d'autre (réponse, objection, commentaire) sous une question, puis Envoyer.

Décisions jusqu'ici : [decisions.md](decisions.md). Ce round porte sur la forme exacte des objets, la structure des kernels fusionnés et l'ordre de travail.

## Q1 : Quel « presque-disque » pour commencer ?

Une densité radiale $\rho(x) = (1 - |x|^2/r^2)_+^{\,k - 1/2}$ a pour projection, **à tout angle**, un polynôme : $f(s) \propto (1 - s^2/r^2)_+^{\,k}$ (transformée d'Abel). Le disque plein est $k = 1/2$, dont la projection est la corde, pas un polynôme.

| k | densité 2D | projection | régularité au bord |
|---|---|---|---|
| 1 | $\sqrt{1-\lvert x\rvert^2/r^2}$ (dôme) | $1 - s^2/r^2$ (degré 2) | $f$ continue, $f'$ saute |
| 2 | $(1-\lvert x\rvert^2/r^2)^{3/2}$ | $(1 - s^2/r^2)^2$ (degré 4) | $f$ et $f'$ continues |

Le $k = 1$ est le plus proche du disque, et le moins cher (primitive de degré 3). Le $k = 2$ donne une loss plus lisse, ce qui aide L-BFGS, mais il est moins « disque ».

➡️ **Recommandé :** famille paramétrée par k (entier), avec un rayon par objet dès le départ (ça ne coûte rien dans le dépôt). On commence par $k = 1$, et on compare $k = 1$ et $k = 2$ sur la convergence L-BFGS. Les diracs sont le cas $r = 0$.

**Réponse :**

super

---

## Q2 : Structure des kernels fusionnés : deux passes sur les points, rien en $[A, N]$ ?

Une évaluation valeur + gradient se ferait ainsi :

1. **Projection + dépôt** (un lancement pour tous les angles) : chaque thread prend un objet, boucle sur les angles, projette et dépose ses masses exactes dans l'histogramme $H[A, M]$. Les atomics passent par des histogrammes privés en mémoire partagée, par tuile d'angles.
2. **Transport par angle** sur $H$ et le sinogramme (sommes cumulées, O(A·M)). On obtient la loss (une réduction) et la dérivée du potentiel $\varphi'[A, M]$, ou mieux sa primitive, pour que l'étape 3 soit exacte.
3. **Projection + gradient** (un lancement) : chaque thread prend un objet, boucle sur les angles, intègre $\varphi'$ contre sa forme par les primitives, et **accumule son gradient en registres**. Une seule écriture de N × 2 floats, sans atomics.

Mémoire : positions et rayons (N × 3), gradient (N × 2), $H$ et $\varphi$ ($A \times M$ ; 600 × 32 768 floats = 80 Mo). Jamais de $[A, N]$. La projection est recalculée aux passes 1 et 3, ce qui coûte moins cher que de la stocker.

➡️ **Recommandé :** oui, exactement cette structure, en CUDA. La version jax pur suit les mêmes trois étapes (par paquets d'angles, pour ne pas matérialiser $[A, N]$), et sert de référence.

**Réponse :**

---

## Q3 : Résolution de la grille fine M ?

La grille fine sert au dépôt ; le sinogramme est constant par bin (B bins). L'approximation porte sur le transport, qui suppose la densité uniforme dans chaque cellule. Pour des objets de rayon ≥ une cellule, la masse déposée est exacte et l'erreur est petite ; pour des diracs, elle est bornée par la masse d'une cellule.

➡️ **Recommandé :** M = s × B avec s entier (par défaut s = 8, soit 16 384 cellules pour 2048 bins), réglable ; on le valide contre le tri exact (écart de loss et de gradient en fonction de s), et on le fige ensuite.

**Réponse :**

---

## Q4 : CPU : kernels C++ loom avec la même structure ?

Tu as demandé du rapide aussi sur CPU. Le jax pur est ~50× plus lent que le C++ sur CPU (mesuré sur le tri).

➡️ **Recommandé :** oui, mêmes trois étapes en C++ loom (multi-thread sur les objets ; histogrammes privés par thread, réduits ensuite), avec un code CPU séparé du CUDA, puisque la vitesse prime sur le partage de code. Il viendra **après** le CUDA.

**Réponse :**

---

## Q5 : Ordre de travail ?

Proposition :

1. Banc errand reproductible (1e5 → 1e7 points, 600 angles, 2048 bins) sur la 2080 Ti et le CPU Mac, qui chronomètre une évaluation valeur + gradient complète (pas étape par étape).
2. Référence jax pur dans sdot : objets « presque-disques » / diracs, dépôt, transport sur grille, gradient ; plus le tri d'événements exact pour valider, et des tests.
3. Optimiseur L-BFGS sur le device dans otrec, et nouvelle API `Reconstruction` (jax seul), en portant les scénarios des tests actuels (disque, chaîne diracs → presque-disques, multi-échelle).
4. Kernels CUDA fusionnés (Q2) branchés via loom/XLA FFI.
5. Axe de lot + `shard_map` sur les angles.
6. Kernels C++ CPU.
7. Suppression de `HcReconstruction/`, `dirac_fused.py` et des branches numpy/torch.

➡️ **Recommandé :** cet ordre. Le premier livrable vérifiable est la fin de l'étape 3 : une reconstruction complète sur le device, en jax pur, avec les tests actuels portés.

**Réponse :**

---

## Q6 : Les bugs torch et le pointer-FFI GPU : on les laisse ?

Les tests torch échouent (CPU : tableaux numpy dans `Sinogram` ; GPU : accès mémoire illégal du kernel `SdotPlan1d`, sans doute le bug pointer-FFI connu de loom). Avec jax seul dans otrec, ces bugs ne nous bloquent plus. Mais le bug GPU est dans loom, et les kernels CUDA de l'étape 4 passeront par le même mécanisme FFI (côté jax, où il marche).

➡️ **Recommandé :** ne pas les traiter dans ce chantier. Ce sera une session à part sur loom (pointer-FFI torch/cupy GPU), que je te propose en fin de grill.

**Réponse :**

---
