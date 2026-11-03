# Reconstruction rapide (otrec) : round 2

> Laisse **Réponse** vide pour accepter la recommandation. `?` pour plus de détails.
> Écris n'importe quoi d'autre (réponse, objection, commentaire) sous une question, puis Envoyer.

Décisions du round 1 : [decisions.md](decisions.md). Le choix du framework attend encore les mesures (tests + timings par framework/device) : il viendra au round 3. Ce round porte sur **l'algorithme** à 1e7 points × 600 angles, qui conditionne tout le reste.

Ordre de grandeur : 1e7 × 600 = 6e9 projections par évaluation. En float32, la matrice `[A, N]` pèse 24 Go, donc on ne la matérialise pas : on traite les angles par paquets. Un tri GPU (radix, cub) de 1e7 clés prend ~5–10 ms, soit **3–6 s par évaluation** rien que pour trier les 600 angles. Le tri est le poste à abattre.

## Q1 : Combien de bins sur le détecteur, et quelle géométrie ?

Le coût du transport côté cible est en O(B) par angle, négligeable devant N si B ≪ N ; mais la valeur de B fixe la résolution des histogrammes de Q3.

➡️ **Recommandé :** faisceau parallèle, angles uniformes sur $[0, \pi)$, B de l'ordre de 1000 à 4000 bins (je dimensionne pour B = 2048, sans le figer dans le code).

**Réponse :**

oui

---

## Q2 : Disques : rayon commun, et calcul sans tri par convolution ?

Avec un rayon $r$ commun, la projection d'un nuage de disques à l'angle $\theta$ est l'histogramme des projections des centres convolé par le profil de corde $c(s) = 2\sqrt{r^2 - s^2}$. On peut donc :

1. déposer les N centres projetés sur une grille fine (dépôt linéaire, O(N), atomics ou tri par paquets) ;
2. convoler par $c$ (FFT ou primitive analytique, O(B log B)) ;
3. faire le transport 1D **densité ↔ densité** sur la grille (fonctions de répartition cumulées, O(B)) ; on obtient le potentiel de Kantorovich $\varphi$ ;
4. gradient : $\partial L / \partial x_i = (\varphi' * c)(\langle x_i, n_\theta \rangle)\, n_\theta$, une interpolation sur la grille, O(N).

Le coût est en O(N + B log B) par angle, **sans aucun tri**, et facile à répartir entre GPU (on somme des histogrammes). Aujourd'hui `DiskModel` fait l'inverse : les bins jouent le rôle des diracs, face à une image des disques rastérisée.

➡️ **Recommandé :** oui. Rayon commun (un rayon par point s'ajoutera plus tard en déposant par classes de rayon), et l'algorithme sans tri ci-dessus pour les disques.

**Réponse :**

On aura a terme des rayons différents, voire des formes différentes. Je serais assez favorable à travailler sur des intervalles, par exemple tri des événements "disque entrant" et " disque sortant".

---

## Q3 : Diracs : tri exact par angle, ou rang par histogramme fin ?

Pour des diracs de même masse face à la densité du sinogramme, le transport exact envoie le dirac de rang $k$ vers le quantile $(k + 1/2)/N$ de la cible : il faut le **rang** de chaque dirac, donc un tri.

- (a) **Tri exact** à chaque évaluation (radix GPU, ~3–6 s/éval à la taille cible), éventuellement accéléré par le fait que les points bougent peu (on repart de l'ordre précédent ; mais garder 600 permutations de 1e7 entiers coûte 24 Go).
- (b) **Rang par histogramme** : on dépose les projections sur une grille fine (M ≫ B cellules), une somme cumulée donne le rang de chaque cellule, et à l'intérieur d'une cellule on répartit les rangs linéairement selon la position. L'erreur sur le rang est bornée par la masse d'une cellule. C'est exactement le chemin des disques avec $r \to 0$ : **un seul code** pour les deux modèles.
- (c) (b) par défaut, (a) en référence pour valider et pour les petits problèmes.

➡️ **Recommandé :** (c). On valide (b) contre (a) sur des tailles moyennes (écart de loss et de reconstruction finale), et on choisit M pour que l'erreur soit sous le bruit du sinogramme.

**Réponse :**

c

---

## Q4 : Reconstructions indépendantes : les grouper sur un même GPU ?

Si plusieurs reconstructions partagent la même géométrie (mêmes angles, bins et N), on peut ajouter un axe de lot à tous les kernels : un seul lancement pour K problèmes. Sinon, on lance un process par GPU et par reconstruction.

➡️ **Recommandé :** les deux. Un axe de lot optionnel (même géométrie) pour remplir un GPU quand N est petit, et un process par GPU pour les gros problèmes. Pas d'ordonnanceur de cluster dans otrec : un script (ou errand) distribue les jobs.

**Réponse :**

oui

---

## Q5 : Quel optimiseur sur le device ?

L'optimiseur d'aujourd'hui est scipy L-BFGS-B, avec nos critères `ftol`, `min_iter` et `disp_tol` (les disques ont des directions plates).

- (a) L-BFGS compilé sur le device (mémoire m ≈ 10 : 2 × 10 × 1e7 × 2 floats = 1,6 Go, ça passe), avec une recherche linéaire qui réutilise **une seule évaluation valeur + gradient** par essai ;
- (b) un optimiseur du premier ordre (Adam, ou descente à pas de Barzilai-Borwein) : moins de mémoire et pas de recherche linéaire, mais beaucoup plus d'itérations ;
- (c) (a) par défaut, (b) disponible.

➡️ **Recommandé :** (c), avec nos critères d'arrêt (`ftol`, `min_iter`, `disp_tol`) réimplémentés sur le device. Le multi-échelle (`split` 50 → 200 → …) est conservé, avec une compilation par étape (formes fixes dans chaque étape).

**Réponse :**

c

---

## Q6 : Que sort-on pendant l'optimisation ?

À 1e7 points, un instantané pèse 80 Mo, et une évaluation de la loss pour un callback coûte autant qu'une itération.

➡️ **Recommandé :** l'historique (loss, nombre de pas, temps) est calculé sur le device et rapatrié en fin d'étape ; des instantanés tous les k pas, optionnels, copiés vers l'hôte de façon asynchrone ; plus de `callback(step, pts)` à chaque pas.

**Réponse :**

---
