# Reconstruction rapide (otrec) : round 3

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour plus de détails.
> Écris n'importe quoi d'autre (réponse, objection, commentaire) sous une question, puis Envoyer.

Décisions jusqu'ici : [decisions.md](decisions.md).

**Les mesures sont arrivées.** En résumé :

| 1e5 points × 180 angles | une évaluation | une itération L-BFGS |
|---|---|---|
| jax pur, CPU Mac | 3,6 s | 7 s |
| kernel C++ `SdotPlan1d` (numpy), CPU | 55 ms (loss seule) | – |
| kernel `dirac_fused`, CPU | 112 ms | 123 ms |
| jax, GPU 2080 Ti | ~105 ms | 240 ms |
| kernel `dirac_fused`, GPU | 673 ms (un thread par angle) | – |

- jax : tous les tests passent, sur CPU et sur GPU (lmo, 2080 Ti).
- torch : tout casse. Sur CPU, le `Sinogram` garde des tableaux numpy (`Sinogram.py:130`) et `torch.compile` échoue sur le gradient. Sur GPU, le kernel CUDA fait un accès mémoire illégal (sans doute le bug pointer-FFI déjà connu).
- Dans le chemin jax, le tri par angle prend 60 à 80 % du temps, et les angles passent un par un (`lax.map`). Le L-BFGS fait deux passes (loss puis gradient), et chaque `run()` recompile (0,3–1 s sur CPU, 3–4 s sur GPU).
- Extrapolé à 1e7 × 600 : plusieurs minutes par itération. Il faut changer d'algorithme, pas seulement optimiser.

## Q1 : Intervalles : tri d'événements, ou dépôt exact des masses par primitives ?

Tu préfères raisonner par intervalles (rayons et formes variables). Deux façons de le faire :

- (a) **Tri des événements** « entrant » ($p_i - r_i$) et « sortant » ($p_i + r_i$), puis balayage. C'est exact, mais il y a 2N clés à trier par angle (deux fois le tri des diracs, déjà le poste dominant). Le balayage n'est simple que si le profil est **polynomial** sur son support : on cumule alors les coefficients des intervalles actifs. Avec la corde $2\sqrt{r^2-s^2}$, la somme des profils actifs n'a pas de forme fermée additive, et le balayage redevient O(N × actifs) ; il faudrait remplacer la corde par un polynôme (ou découper chaque forme en morceaux polynomiaux).
- (b) **Dépôt exact sur une grille fine**, sans tri. Chaque intervalle dépose dans chaque cellule $[s_j, s_{j+1}]$ qu'il touche sa masse exacte $F_i(s_{j+1}) - F_i(s_j)$, où $F_i$ est la primitive de sa forme (corde, polynôme, n'importe quelle forme dont on sait la primitive). Coût : O(N × cellules couvertes) en atomics ; c'est ~2 à 3 cellules par disque quand le rayon est de l'ordre d'une cellule. On fait ensuite le transport densité ↔ densité sur la grille (sommes cumulées, O(M)), puis le gradient de chaque intervalle s'obtient par les mêmes primitives : $\partial L/\partial p_i = \int \varphi'(s)\, f_i(s - p_i)\, ds$.
  Les diracs sont le cas $r \to 0$ (Q3 du round 2) : **un seul chemin de code** pour diracs, disques et formes.
- (c) (b) par défaut, (a) en référence exacte pour valider (avec un profil polynomial, ou en petite taille).

➡️ **Recommandé :** (c). (b) garde l'idée d'intervalles (chaque objet est un support avec une primitive), sans payer de tri, et s'adapte naturellement à plusieurs GPU : on somme des histogrammes. La seule approximation est la résolution M de la grille, que l'on valide contre (a).

**Réponse :**

Oui, j'avais en tête aussi des formes dont la projection sont des polynômes. Il y aura un jeu sur leur choix. On pourrait commencer par des "presque-disques" pour démarrer

---

## Q2 : Un seul framework : jax ?

- **jax** : tous les tests passent, sur CPU et GPU. On a `jit` de bout en bout, `lax.while_loop` pour l'optimiseur sur le device, `vmap` pour l'axe de lot, `shard_map` (avec un `psum`) pour répartir les angles entre GPU, et `jax.distributed` pour plusieurs nœuds. Les kernels loom passent par XLA FFI.
- **torch** : aujourd'hui cassé (CPU et GPU), et `torch.compile` gère mal une boucle d'optimisation complète. Le multi-GPU passerait par `torch.distributed`, plus manuel.

➡️ **Recommandé :** jax, et uniquement jax dans otrec (plus de branches numpy/torch). Le CPU est servi par jax + kernels loom C++ via FFI, sans chemin numpy séparé.

**Réponse :**

On peut se focaliser sur Jax, mais il est certain qu'on aura la meilleurs efficacité en faisant un certain nombre de kernels CUDA

---

## Q3 : Kernels : jax pur d'abord, puis loom (C++/CUDA) là où la mesure l'exige ?

Le dépôt (scatter des masses), la somme cumulée et le gradient (gather) s'écrivent en jax pur (`segment_sum`, `cumsum`, `take`). Sur CPU, XLA est lent pour ce genre de boucles (le C++ `SdotPlan1d` est 50× plus rapide que le jax pur sur le tri) ; sur GPU, le scatter-add d'XLA est correct mais pas fusionné avec la projection.

➡️ **Recommandé :** écrire d'abord le chemin en jax pur (référence, et il tourne partout), mesurer, puis remplacer par des kernels loom (un C++ CPU et un CUDA, séparés si c'est plus rapide) les étapes qui dominent. On s'attend à ce que ce soit le dépôt et le gradient fusionnés avec la projection, sur tous les angles en un seul lancement.

**Réponse :**

Oui mais il faut faire attention aux "étapes" qui sont mesurées indépendamment alors que fused elles peuvent être beaucoup plus rapides et consommer beaucoup moins de mémoire.

---

## Q4 : Multi-GPU : angles répartis, points répliqués ?

Avec 4 GPU sur une reconstruction : chaque GPU traite 150 angles. Les positions (1e7 × 2 floats = 80 Mo) et l'état L-BFGS sont **répliqués**, et le gradient est sommé par un all-reduce de 80 Mo par évaluation (~3–10 ms en NVLink, plus en PCIe). Variante : répartir aussi les points (chaque GPU ne garde qu'un quart de l'état L-BFGS), mais ça complique tout.

➡️ **Recommandé :** angles répartis, points et optimiseur répliqués, avec `shard_map` sur un axe « angles ». Le même code tourne sur 1 GPU (axe de taille 1).

**Réponse :**

---

## Q5 : Où vit le code ?

Le transport 1D « somme d'intervalles à primitive connue ↔ densité sur une grille » est générique : il a sa place dans sdot, à côté de `SdotPlan1d`. La boucle de reconstruction, l'optimiseur sur le device et le multi-échelle restent dans otrec. Par ailleurs, `HcReconstruction/` (jax pur, séparé) et `dirac_fused.py` (kernel CPU, un thread par angle) font doublon avec le nouveau chemin.

➡️ **Recommandé :** kernels et transport 1D dans sdot ; otrec réécrit autour, en jax seul. `HcReconstruction/` et `dirac_fused.py` sont supprimés une fois le nouveau chemin validé (leurs modèles polygones et disques se reformulent comme des formes à primitive connue).

**Réponse :**

---

## Q6 : Sur quel matériel juge-t-on la vitesse, et avec quel objectif chiffré ?

Le seul GPU accessible ici est la 2080 Ti de lmo (11 Go, pas de NVLink). À 1e7 points, l'état L-BFGS (~1,6 Go) et les points tiennent, mais pas un `[A, N]`.

➡️ **Recommandé :** développement et mesures sur la 2080 Ti (1 GPU) et le CPU Mac, avec une cible de **< 1 s par évaluation valeur + gradient à 1e7 × 600 × 2048 sur un GPU**, et un banc reproductible (errand) qui sort les temps par étape (dépôt, transport, gradient, optimiseur). Le multi-GPU est validé sur la machine du cluster que tu indiqueras (laquelle ?).

**Réponse :**

On veut être parcimonieux en mémoire. A priori tout passe sur la 2080 pour l'instant. On ira sur d'autres GPUs à un moment mais on reste là pour l'instant

---
