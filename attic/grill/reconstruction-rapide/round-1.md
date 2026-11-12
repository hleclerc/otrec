# Reconstruction rapide (otrec) : round 1

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour plus de détails.
> Écris n'importe quoi d'autre (réponse, objection, commentaire) sous une question, puis Envoyer (ou sauve le fichier et dis « ok » dans le chat).

Contexte : deux agents tournent en parallèle (cartographie du chemin loss/grad jusqu'à sdot/loom, et tests + timings par framework/device). Les questions qui dépendent de leurs résultats (quel framework, où passe le temps, quels kernels réécrire) viendront au round 2. Ce round fixe les **objectifs**.

## Q1 : Quel périmètre de modèles ?

otrec a plusieurs chemins : diracs 2D (`DiracModel`, transport 1D par angle), disques (`DiskModel`, loss lisse), diracs 3D (`ProjectedDiracModel`, SdotPlanNd 2D par angle), `halo`, `mesh`, et un `HcReconstruction/` séparé en jax.

Options :
- (a) diracs 2D seulement (transport 1D) ;
- (b) diracs 2D + enchaînement diracs → disques (le cas d'usage de `test_chain_diracs_then_disks`) ;
- (c) tout, y compris la 3D.

➡️ **Recommandé :** (b). Le transport 1D est le cœur visé ; les disques restent dans la même boucle d'optimisation et doivent suivre le même framework. La 3D (SdotPlanNd) et `HcReconstruction` sont un chantier distinct, à traiter après.

**Réponse :**

Le 3D projeté 2D est l'objectif final, mais on a besoin du 2D projeté 1D en première étape. On peut déjà se concentrer sur le 1D

---

## Q2 : Quelle taille de problème vise-t-on ?

Les tests tournent à 80–400 points, 8–24 angles, ~100–200 bins : là, tout est dominé par l'overhead Python. Le « très rapide » n'a pas le même sens selon l'échelle :

| régime | N points | angles | bins | ce qui compte |
|---|---|---|---|---|
| petit | 1e2–1e3 | 8–32 | 128–512 | latence par itération (overhead, lancements de kernels) |
| moyen | 1e4–1e5 | 64–180 | 512–2048 | débit d'un seul GPU |
| grand | 1e6–1e8 | 180–1000+ | 2048–8192 | mémoire, multi-GPU |

➡️ **Recommandé :** optimiser pour **moyen → grand** (cible de référence : N = 1e6, 360 angles, 2048 bins, en float32), en gardant le petit régime correct mais sans chercher à battre le CPU sur les tests.

**Réponse :**

Pour le 2D projeté 1D, 1e7 diracs ou disques, 600 angles.

---

## Q3 : « Grande échelle » = un gros problème, ou beaucoup de problèmes ?

- (a) **un seul problème** trop gros pour un GPU : on répartit les **angles** entre GPU (chaque angle = un transport 1D indépendant ; seul le gradient sur les positions est sommé, un all-reduce de taille N×2 par itération) ;
- (b) **beaucoup de reconstructions indépendantes** (lots de sinogrammes, balayage de paramètres) : parallélisme trivial, un job par GPU ;
- (c) les deux.

➡️ **Recommandé :** (c), mais concevoir pour (a) : le découpage par angles avec all-reduce du gradient. (b) en découle gratuitement (vmap / un process par GPU).

**Réponse :**

Pour le 1D, on aura beaucoup de reconstructions indépendantes, mais si on a 4 GPUs, ça vaudra le coup de faire les angles en parallèle.

---

## Q4 : La boucle d'optimisation doit-elle tourner entièrement sur le device ?

Aujourd'hui l'optimiseur (`optimizers.py` : L-BFGS, descente de gradient) est piloté côté Python ; si c'est scipy L-BFGS-B, chaque itération fait un aller-retour hôte ↔ GPU et une synchronisation.

- (a) garder l'optimiseur côté hôte (simple, mais plafonne la vitesse aux petites et moyennes tailles) ;
- (b) optimiseur entièrement sur le device et compilé (ex. L-BFGS optax/jax sous `jit`, `lax.while_loop`), callbacks seulement tous les k pas ;
- (c) les deux, (b) par défaut.

➡️ **Recommandé :** (c). Sans (b), aucun gain sur les kernels ne se voit en dessous de ~1e5 points. On garde (a) pour le débogage et les comparaisons (`min_iter`, `disp_tol` du test disques dépendent du comportement scipy).

**Réponse :**

Il faut que ça calcule vite, donc rester sur le device me paraît utile.

---

## Q5 : Précision : float32 accepté sur GPU ?

Le transport 1D par tri est stable en float32 pour les positions, mais les sommes cumulées de masses sur 2048+ bins et 1e6+ points perdent des chiffres ; les tests disques visent des tolérances comme `ftol = 1e-14` et « 1.05 × plancher ».

➡️ **Recommandé :** float32 par défaut pour les gros calculs, avec sommes cumulées compensées (ou par blocs) dans le kernel ; float64 en option (et dans les tests de précision).

**Réponse :**

Oui

---

## Q6 : A-t-on le droit de changer l'API de `Reconstruction` ?

Aller vite peut demander de figer des formes (nombre de points par étape, buckets), de tout passer en tableaux du framework choisi (plus de `np.asarray` à chaque callback), et de remplacer `callback(step, pts)` par un callback espacé.

➡️ **Recommandé :** oui, quand c'est nécessaire, en gardant l'API actuelle des tests (`diracs()`, `disks()`, `multiscale()`, `history`, `frames`) et en adaptant les tests si un comportement change volontairement.

**Réponse :**

On peut tout changer. Ce qu'il y a dans OtRec est juste un premier jet pour tester.

---
