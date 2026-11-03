# Reconstruction 3D : round 2 (transport 2D exact par vue)

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour demander plus de détails.

**Ce que sdot sait faire sur GPU** (lu dans le code, pas encore mesuré sur ce cas) :

- Diracs → `Image` 2D : tout le Newton tient en **un seul appel ffi** (`OtProblem(...).solve(Iterative(...))`), sous `jax.jit`, avec un arbre construit sur la carte.
- Warm start avec `weights0=` : les poids sont rendus par le plan. Si les poids donnés vident une cellule, le C++ choisit le meilleur départ.
- Retours : `barycenters`, `cell_masses`, `cost`, `cost_and_position_grad()` = $2m_i(p_i - b_i)$.
- `differentiable=False` évite les diagrammes en plus et le solve laplacien quand seul ce gradient sert.
- Sous jit, le statut n'est pas vérifié : c'est à nous de le lire.
- Pas de vmap sur des problèmes indépendants : un solve par vue, ce qui est cohérent avec « un angle à la fois par carte ».
- `lax.map` / `while_loop` autour du solve : non testé. Le handler fait de toute façon des allers-retours hôte à chaque diagramme.
- Chiffres enregistrés (départ à froid, 2080 Ti) : image 512², 2e4 diracs en 0,83 s, 1e5 diracs en 1,9 s. Le temps est dominé par les solveurs linéaires. Aucun chiffre en warm start.
- Limites : image sur grille régulière alignée sur les axes (ça va avec tout détecteur plan, conique compris), pas de périodicité.

---

## Q1 : taille de l'exploration ?

L'idée est d'observer la dynamique, donc de lancer beaucoup d'essais. Une évaluation coûte A solves warm start (inconnu, peut-être 0,05–0,3 s chacun à 1e5).

➡️ **Recommandé :** démarrer à **1e4–1e5 diracs, 30–100 vues, détecteur 256²**, puis monter (1e6, 512², puis plus) quand la dynamique est comprise. La première mesure du round sera le coût d'un solve warm start en fonction de N et de la taille d'image.

**Réponse :**

---

## Q2 : quel pas d'optimisation ?

Pour chaque vue k, le transport donne le barycentre $b_{k,i}$ de la cellule de chaque dirac sur le détecteur. Le gradient est $\sum_k J_k(x_i)^T\, 2m_i\,(P_k(x_i) - b_{k,i})$.

- **(a) L-BFGS** sur $\sum_k W_2^2$ (l'optimiseur sur le device du 2D). Chaque essai de la recherche linéaire coûte A solves, et les diracs rendent la perte non lisse (c'est la remarque de Q6 au round 1), ce qui peut gêner L-BFGS.
- **(b) Pas de « Lloyd relevé »** : chaque dirac va au point 3D dont les projections collent au mieux aux barycentres,
  $$x_i \leftarrow \arg\min_x \sum_k |P_k(x) - b_{k,i}|^2,$$
  - en faisceau parallèle, c'est un système 3×3 par dirac ;
  - en conique, c'est un pas de Gauss-Newton sur les jacobiennes : on ne fait pas d'hypothèse « parallèle » ;
  - le pas est relâché par τ ∈ (0, 1] ;
  - c'est **une évaluation par pas**, sans recherche linéaire, et l'analogue direct de Lloyd en 2D (qui converge bien malgré la non-différentiabilité).
- **(c) Les deux**, à comparer sur les mêmes cas.

➡️ **Recommandé :** (c), avec (b) comme outil principal pour l'observation : chaque pas a un sens géométrique clair, et le désaccord entre les vues se mesure directement (voir Q5).

**Réponse :**

---

## Q3 : où vivent les poids du warm start ?

Ce sont A × N poids : 100 × 1e5 en float64 font 80 Mo, mais 600 × 1e7 en feraient 48 Go. Un départ un peu périmé reste valide, car le C++ choisit le meilleur départ.

- **(a)** Sur le device tant que ça tient, sur l'hôte sinon, avec une copie asynchrone par vue.
- **(b)** Toujours sur l'hôte.
- **(c)** Pas de warm start : plus simple, mais plusieurs fois plus lent.

➡️ **Recommandé :** (a).

**Réponse :**

a

---

## Q4 : structure de la boucle et multi-GPU ?

Le solve est un appel ffi par vue qui fait déjà des allers-retours hôte. Le gain d'un programme entièrement compilé (`lax.map` sur les vues dans un `while_loop`) serait faible, et ce n'est pas testé.

- **(a)** Une boucle Python sur les pas et les vues, avec un solve jitté par vue et un pas jitté (relèvement + mise à jour). En multi-GPU, les vues sont réparties entre les cartes (vue k sur la carte k mod n_gpu) avec dispatch asynchrone : chaque carte avance ses vues, puis on somme. Un seul process.
- **(b)** Tout dans un programme jax (`shard_map` sur les vues, `lax.map` à l'intérieur).

➡️ **Recommandé :** (a). Le surcoût Python est négligeable devant un solve de l'ordre de 0,1 s. On garde (b) en tête si le solve devient très court.

**Réponse :**

---

## Q5 : que mesurer et regarder ?

Il s'agit de comprendre comment les diracs se déplacent et comment ça converge.

À chaque pas, sur le device puis rapatrié :

- le coût total et le coût par vue ;
- les itérations et diagrammes de Newton, et le statut, par vue ;
- le déplacement (moyenne, max) ;
- **le désaccord entre vues** : le résidu du relèvement de Q2 par dirac, $\sum_k|P_k(x_i)-b_{k,i}|^2$ au point relevé. Il indique les diracs « tiraillés » entre vues incompatibles.

Sur un fantôme connu :

- la distance à la vérité (fraction des diracs dans les boules, W2 à un échantillon de la vérité).

Instantanés des positions tous les k pas (npz).

Vues :

- nuage 3D coloré par le désaccord ;
- par vue, les points projetés sur la radiographie avec leurs cellules (petits cas) ;
- courbes de convergence.

➡️ **Recommandé :** tout ceci, avec un `experiment` errand qui produit les npz et une page de visualisation.

**Réponse :**

Oui très bien. Il faudrait aussi afficher si les points auraient intérêt à grossir ou pas (changement du poids) pour minimiser le coût (je sais, ça fait un gradient en plus à calculer mais ça m'intéresse beaucoup).

---

## Q6 : point de départ et fantômes ?

- **Départ** : l'enveloppe visuelle (`visual_hull_points`, qui existe), ou un tirage uniforme dans la boîte.
- **Fantômes** : quelques boules (`add_sphere`), puis des objets plus durs (boules creuses, deux boules qui se touchent, structures fines), avec un fond de 1e-6 comme dans les notes.

➡️ **Recommandé :** les deux départs (comparer), l'enveloppe visuelle par défaut, et les fantômes de boules d'abord, puis la liste plus dure.

**Réponse :**

---

## Q7 : géométrie générique ?

Pour ne pas figer le parallèle, on introduit une **`Geometry`** qui porte, par vue :

- `project(x) → (u, v)` en coordonnées du détecteur ;
- la jacobienne (pour le gradient et le relèvement de Q2) ;
- le cadre du détecteur (taille en pixels, extent).

Il y en aurait deux implémentations :

- `ParallelGeometry(bases [A, 2, 3])`, maintenant ;
- `ConeGeometry` (source, détecteur), plus tard.

`Radiographs` porte une `Geometry` et les images.

➡️ **Recommandé :** oui, en jax.

**Réponse :**

---

## Q8 : ordre de travail ?

1. `Geometry` (parallèle) + `Radiographs` en jax, avec le fantôme de boules.
2. Banc errand : un solve par vue, à froid et à chaud, selon N et la taille d'image, sur la 2080 Ti.
3. `Reconstruction3d` : boucle de Q4, pas de Lloyd relevé et L-BFGS, mesures de Q5.
4. Expériences sur les fantômes et page de visualisation.
5. Répartition des vues entre GPU (testée avec 4 devices CPU, puis sur une machine multi-GPU quand on en aura une).

➡️ **Recommandé :** cet ordre. On s'arrête après 4 pour regarder les résultats ensemble avant d'aller plus loin.

**Réponse :**

---
