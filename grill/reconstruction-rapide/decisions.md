# Reconstruction rapide (otrec) : décisions

- **Périmètre** : d'abord le 2D projeté en 1D (diracs, et disques). Le 3D projeté en 2D est l'objectif final, pour plus tard.
- **Taille cible (1D)** : 1e7 diracs ou disques, 600 angles.
- **Échelle** : surtout beaucoup de reconstructions indépendantes ; avec 4 GPU, répartir les angles d'une même reconstruction entre GPU en vaut la peine.
- **Optimiseur** : la boucle reste sur le device (pas d'aller-retour hôte par itération).
- **Précision** : float32 accepté sur GPU.
- **API** : on peut tout changer ; l'otrec actuel est un premier jet.
- **Géométrie** : faisceau parallèle, angles uniformes sur [0, π), B ≈ 1000–4000 bins (dimensionné pour 2048, non figé).
- **Disques** : à terme rayons et formes différents ; Hugo préfère raisonner par intervalles (événements « entrant » / « sortant »). Algorithme exact à trancher au round 3.
- **Diracs** : rang par histogramme fin par défaut, tri exact en référence (validation, petits problèmes).
- **Lots** : axe de lot optionnel (même géométrie) sur un GPU + un process par GPU pour les gros problèmes ; pas d'ordonnanceur dans otrec.
- **Optimiseur** : L-BFGS compilé sur le device par défaut (une évaluation valeur + gradient par essai), premier ordre disponible ; critères `ftol`/`min_iter`/`disp_tol` sur le device ; multi-échelle gardé, une compilation par étape.
- **Sorties** : historique sur le device, rapatrié en fin d'étape ; instantanés optionnels tous les k pas, copie asynchrone ; plus de callback à chaque pas.
- **Algorithme 1D** : objets = supports à primitive connue, dépôt exact des masses sur une grille fine (sans tri) par défaut ; tri d'événements en référence. Formes dont la projection est polynomiale (choix de la forme = un paramètre à explorer), en commençant par des « presque-disques ».
- **Framework** : jax seul dans otrec ; la meilleure efficacité viendra de kernels CUDA (via loom).
- **Kernels** : jax pur d'abord comme référence, puis loom. Attention : ne pas juger étape par étape, la fusion (projection + dépôt, projection + gradient) peut être bien plus rapide et beaucoup plus économe en mémoire.
- **Multi-GPU** : angles répartis (`shard_map`), points et optimiseur répliqués, all-reduce du gradient ; même code sur 1 GPU.
- **Code** : transport 1D et kernels dans sdot ; otrec réécrit en jax seul ; `HcReconstruction/` et `dirac_fused.py` supprimés une fois le nouveau chemin validé.
- **Matériel** : la 2080 Ti (11 Go) pour l'instant ; être parcimonieux en mémoire.
- **Forme de départ** : famille de densités radiales $(1-|x|^2/r^2)_+^{k-1/2}$, de projection $(1-s^2/r^2)_+^k$, avec un rayon par objet ; on commence par k = 1 et on compare k = 1 et k = 2 sur la convergence L-BFGS ; les diracs sont le cas r = 0.
- **Kernels fusionnés** : (1) projection + dépôt dans H[A, M], tous les angles en un lancement, histogrammes privés en mémoire partagée ; (2) transport par angle → loss + potentiel ; (3) projection + gradient, accumulé en registres, sans atomics. Jamais de [A, N]. La référence jax pur suit les mêmes étapes, par paquets d'angles.
- **Grille fine** : M = s × B, s = 8 par défaut, réglable ; validé contre le tri exact, puis figé.
- **CPU** : mêmes trois étapes en C++ loom (multi-thread, histogrammes privés par thread), code séparé du CUDA, après le CUDA.
- **Ordre** : (1) banc errand d'une évaluation complète ; (2) référence jax pur dans sdot + tri d'événements exact + tests ; (3) L-BFGS sur le device + nouvelle API `Reconstruction`, tests portés (premier livrable) ; (4) kernels CUDA fusionnés ; (5) axe de lot + `shard_map` ; (6) kernels C++ CPU ; (7) suppression de `HcReconstruction/`, `dirac_fused.py`, branches numpy/torch.
- **Bugs torch / pointer-FFI GPU** : hors de ce chantier ; session loom à part.

## Réalisé ( 2026-10-17, non commité )

- Étapes 1 à 6 faites : banc `otrec/bench/bench_eval.py`, `sdot.projected1d` ( + kernels `sdot/Projected1d.h` ), `Reconstruction2d` + `device_optim`, lots, `shard_map`, kernels CPU.
- 2080 Ti, une évaluation valeur + gradient à 1e7 × 600 × 2048 : 615 ms, 688 Mo ( l'ancien chemin : ~minutes ).
- Écarts au plan : ordre Morton des objets ( indispensable, 20× sur le dépôt ) ; transport sur grille en float64 ( float32 : gradient faux à 47 % ) ; jax x64 activé par `reconstruction2d`.
- Étape 7 ( suppressions ) en attente : `lung_alveoli` et `viz/line_search_compare.py` utilisent encore `dirac_fused` ; l'ancien `Reconstruction` sert encore la 3D et `halo`.
- Bilan perf 2D ( 2026-10-17 ) : à 1e7, dépôt 385 ms ≈ 6e10 atomics globaux ( ~150 G/s, limite L2 de la 2080 Ti ), grille 17 ms, gather 219 ms. Essai de dépôt privatisé en mémoire partagée : plus lent ( 615 ms : atomicAdd float en shared = boucle CAS sur Turing ), abandonné. Le coût suit N × A × L ( L = cellules par objet ) : on est proche du maximum pour cette formulation ; on passe à la 3D.
