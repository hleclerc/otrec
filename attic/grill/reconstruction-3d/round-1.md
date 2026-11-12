# Reconstruction 3D : round 1

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour demander plus de détails.
> Écris autre chose (réponse, objection, commentaire) sous n'importe quelle question, puis Envoyer.

**Contexte.** La 2D est proche du maximum : à 1e7 × 600 × 2048 sur la 2080 Ti, une évaluation prend 615 ms, dont 385 ms de dépôt, limité par le débit des atomics en L2. L'essai de dépôt en mémoire partagée a été plus lent sur Turing, donc il est abandonné.

La 3D actuelle est dans `Radiographs` + `ProjectedDiracModel` + l'ancien `Reconstruction` :

- faisceau parallèle, un seul axe de rotation (z), angles sur [0, π) ;
- détecteur de 40² à 128² pixels, 3 à 100 vues ;
- une perte en **transport 2D semi-discret exact** par vue (sdot, Newton sur diagrammes de puissance), avec une boucle Python sur les vues ;
- optimisation par scipy L-BFGS, côté hôte et sur CPU en pratique ;
- environ « des minutes par pas » à 20 000 diracs et 6 vues.

sdot a un Newton 2D sur GPU, mais pas de lot de problèmes indépendants, et un diagramme à 1e7 graines coûte au moins une seconde. Au total, c'est hors d'échelle pour 1e7 points × des centaines de vues.

---

## Q1 : quelle perte pour comparer une radiographie aux points projetés ?

Une remarque change tout. Projeter la radiographie de la vue θ sur une direction φ **du détecteur** revient à projeter les points 3D sur la direction $n = \cos\varphi\, u_\theta + \sin\varphi\, v$ de $\mathbb{R}^3$. La donnée 1D correspondante, c'est-à-dire les sommes de la radiographie le long des lignes perpendiculaires à φ, est l'intégrale de la densité sur les plans orthogonaux à n.

On retombe donc **exactement** sur le problème 1D qu'on vient d'optimiser : mêmes kernels (dépôt, grille, gather), avec des normales `[T, 3]` au lieu de `[A, 2]`. Avec un seul axe de rotation, toute direction de $S^2$ est accessible depuis une vue, celle où $d_\theta \perp n$.

Options :

- **(a) Sliced W2 par vue.** Chaque vue fournit quelques directions φ, et la perte est la somme des W2² 1D sur toutes les paires (vue, φ). On réutilise tout : kernels, lot, `shard_map`, L-BFGS sur le device.
- **(b) Transport 2D exact par vue**, comme aujourd'hui : il faudrait un Newton 2D en lot sur GPU, ce qui représente un très gros chantier pour un coût d'environ 1 s par vue à 1e7.
- **(c) Les deux** : (a) par défaut, (b) comme référence de validation sur de petits cas avec le code existant.

Le sliced est une distance différente (plus faible) : elle converge vers le même minimum quand les données sont cohérentes, mais le paysage n'est pas le même. Sur de petits cas, c'est (b) qui dira si la qualité de reconstruction tient.

➡️ **Recommandé :** (c). Sliced par défaut, transport 2D exact en référence sur de petits cas, pour comparer la qualité de reconstruction à nombre de pas égal.

**Réponse :**

La carte 2080, c'est juste pour s'entrainer. Après on passera sur des plus gros et plus récents modèles.

Une des objectifs de ce round est justement de voir comment le transport 2D se comporte. L'idée est de voir comment le diracs se déplacent et comment ça converge.

À la fin, l'idée sera sans doute de ne faire de la 2D qu'à partir d'une bonne approximation... mais je voudrais tester et comprendre avant ça.

Rq: pour moi, il n'y a pas d'enjeux autour des batchs de pb de transport 2D. Chaque problème sera long et l'idée pourra être de faire 1 angle à la fois par carte GPU.

---

## Q2 : quel jeu de directions ?

Le coût d'une évaluation est proportionnel à N × T × L, où T est le nombre total de directions 1D. Avec 1e7 points :

- T = 600 donne environ 0,6 s (comme en 2D) ;
- T = 600 vues × 32 φ donnerait environ 20 s.

Options :

- **(a) Budget T global, réparti ~uniformément sur $S^2$**, chaque direction étant rattachée à la vue qui la voit (vue la plus proche de $d \perp n$). Les données 1D se calculent à partir de cette vue. Peu de redondance, couverture isotrope.
- **(b) D directions φ fixes par vue** (par ex. D = 8 régulières). T = A × D, simple, mais redondant : φ = 90° donne la même direction z pour toutes les vues.
- **(c) Directions tirées au hasard à chaque pas** (sliced stochastique). C'est bon marché par pas, mais la perte n'est plus déterministe, ce qui casse L-BFGS (Armijo, ftol) et oblige à repasser en premier ordre.

➡️ **Recommandé :** (a), avec un jeu fixe pendant une étape d'optimisation. On pourra le changer entre étapes (rotation du jeu à chaque étape multi-échelle) pour couvrir davantage sans casser L-BFGS. T est un paramètre (défaut ≈ nombre de vues).

**Réponse :**

---

## Q3 : tailles cibles en 3D ?

À fixer pour dimensionner (mémoire et grille fine) :

- N points (1e7 comme en 2D ? plus ?) ;
- nombre de vues A ;
- taille du détecteur (pixels nu × nv) ;
- nombre de bins B des profils 1D (lié à la diagonale du détecteur) ;
- T directions.

➡️ **Recommandé :** 1e7 points, A ≈ 600 vues, détecteur ≈ 1024², B ≈ 2048 (M = 8 × B = 16 384 cellules fines), T ≈ 600 à 2000. Les données (A radiographies de 1024² en float32) font 2,4 Go, ce qui tient sur la 2080 Ti à côté du reste.

**Réponse :**

On peut faire 1e7 pour tester, mais on passera à plus (sur d'autre matériel)

---

## Q4 : comment calculer les profils 1D mesurés à partir des radiographies ?

Un pixel du détecteur est un carré uniforme. Projeté sur la direction φ, il donne un **trapèze**, dont la masse se dépose exactement sur la grille fine (primitive par morceaux de degré 2). C'est le même schéma que le dépôt des objets, avec un autre profil.

Options :

- **(a) Dépôt exact des pixels en trapèzes, une fois au début, sur le device.** On garde ensuite seulement les profils `[T, M]` et leurs cumuls : quelques centaines de Mo, et les radiographies peuvent quitter le GPU.
- **(b) Rééchantillonnage bilinéaire de l'image le long des lignes** : plus simple, mais approché et sensible à l'aliasing.

➡️ **Recommandé :** (a). C'est un kernel de plus (un « objet » = un pixel, de masse donnée), exact, et la mémoire en régime est faible.

**Réponse :**

Je suis pour (a)... mais pas pour faire de sliced de sliced à cette étape

---

## Q5 : géométrie ?

- **(a) Faisceau parallèle**, avec des bases de vue génériques `[A, 2, 3]` : un axe unique aujourd'hui, mais aussi des séries de tilt ou des orientations quelconques, gratuitement.
- **(b) Faisceau conique** dès maintenant : la projection n'est plus linéaire et l'équivalence de Q1 (plan ↔ ligne du détecteur) ne tient plus exactement.

➡️ **Recommandé :** (a). Le conique est une autre étape, à trancher plus tard.

**Réponse :**

Le conique est une autre étape, mais on va l'avoir à un moment donc il ne faut pas faire trop d'hypothèses basées sur le parallèle

---

## Q6 : formes d'objets en 3D ?

Une boule uniforme de rayon r a des intégrales sur plans proportionnelles à $\pi (r^2 - s^2)$. Sa projection 1D est donc exactement le « dôme » k = 1 de la 2D. Plus généralement, la densité 3D $(1 - |x|^2/r^2)_+^{k-1}$ projette sur $(1 - s^2/r^2)_+^k$.

On garde donc les **mêmes profils 1D** : diracs (r = 0, tente), boules (k = 1), boules adoucies (k = 2), avec un rayon par objet.

➡️ **Recommandé :** oui, mêmes profils, à paramétrer seulement par la dimension (qui change la densité 3D affichée, pas le kernel).

**Réponse :**

oui, même profils, mais je propose pour démarrer de mettre uniquement des diracs. Les dômes n'interviendront que pour finaliser la convergence (les diracs rendent le pb non différentiable)

---

## Q7 : organisation du code ?

- **sdot** : `ProjectedLoss1d` est généralisé à des points de dimension d (normales `[T, d]`) ; les kernels sont templatés sur d (la seule différence est le produit scalaire), plus le dépôt des pixels (Q4).
- **otrec** : `Reconstruction2d` devient `Reconstruction` (dimension tirée des données : `Sinogram` → 2D, `Radiographs` → 3D), avec le même optimiseur, le même lot et le même `shard_map` (sur les directions).
- L'ancien chemin 3D (`ProjectedDiracModel`, ancien `Reconstruction`) reste comme référence (Q1 c) jusqu'à validation, puis il est retiré avec l'étape 7.

➡️ **Recommandé :** comme ci-dessus.

**Réponse :**

---

## Q8 : ordre de travail ?

1. Profils mesurés : dépôt exact des pixels (jax pur puis kernel), test contre la somme directe des pixels.
2. `ProjectedLoss1d` en dimension d, avec tests 3D (gradients par différences finies, lot, sharding).
3. `Reconstruction` unifiée 2D/3D, avec le jeu de directions de Q2 et les tests 3D portés (boules : centres retrouvés).
4. Comparaison sliced vs transport 2D exact sur de petits cas : qualité à pas égal, temps.
5. Banc 3D errand à la taille cible sur la 2080 Ti.

➡️ **Recommandé :** cet ordre ; (4) décide si le sliced suffit.

**Réponse :**

---
