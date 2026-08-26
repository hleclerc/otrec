"""Projection (transformée de Radon) DIFFÉRENTIABLE d'une union de disques de rayon FIXE.

C'est la brique géométrique du modèle `DiskModel` (`models.py`), qu'un `Reconstruction`
(`Reconstruction.py`) fait décroître : le sinogramme mesuré y est vu comme des diracs pondérés,
et le MODÈLE auquel on le compare est l'image constante par morceaux produite ici.

Ce qu'on gagne à modéliser des disques plutôt que des diracs : la perte devient une fonction
LISSE des inconnues. Le coût OT 1D est linéaire en la densité cible ; cette densité est ici
l'image projetée, dont la dépendance aux centres est la corde `2*sqrt(r^2 - u^2)` INTÉGRÉE sur
chaque pixel -- donc C^1 en la position projetée (voir `DiskProjector._values_of`). Il n'y a
plus de terme non lisse hérité du déplacement de diracs.

Conventions géométriques : celles de `Sinogram` (angles theta_k = k*pi/nb_angles, normale
n_k = (cos, sin), coordonnée détecteur s = p.n_k). La grille de l'image modèle couvre la MÊME
étendue que le détecteur, mais peut être plus fine (`nb_pixels`) pour représenter correctement
la projection d'un disque de petit rayon.

Tout est BATCHÉ sur `num_angle` : un seul `OtPlan1d` traite les `nb_angles` transports, et le
chemin pris est celui, purement Jax, de `Image.try_update_otplan1d` (aucun kernel C++).
"""
import numpy as np

from loom import Tensor, driver, RealTensor
from sdot import Image

from .Sinogram import Sinogram


class DiskProjector:
    """Projection différentiable d'une union de disques de RAYON FIXE, discrétisée en une
    fonction constante par morceaux par angle -- l'`Image` batchée que consomme `OtPlan1d`.

    La géométrie (angles, étendue détecteur, grille image, rayon) est fixée à la construction ;
    seuls les CENTRES varient d'un appel à l'autre -- ce sont eux l'inconnue de l'optimisation.

    `nb_pixels` : nombre de cases de la grille image, sur la même étendue que le détecteur (par
    défaut `sinogram.nb_bins`). Le prendre plus grand affine la représentation de la projection
    (utile quand `radius` est petit devant la largeur d'une case détecteur) sans toucher aux
    données mesurées, qui restent échantillonnées sur `nb_bins`.
    """

    def __init__( self, sinogram: Sinogram, radius: float, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24 ) -> None:
        if radius <= 0:
            raise ValueError( "radius doit être > 0" )
        self.sinogram = sinogram
        self.radius = float( radius )
        # comptes côté HÔTE, figés ici (hors de tout trace) : `values` est appelée SOUS le `jit` de
        # l'optimiseur, où relire une ShapeVar donnerait un tracer (cf. `Sinogram.nb_bins_host`).
        self.nb_angles = int( sinogram.nb_angles.value )
        self.nb_pixels = int( nb_pixels if nb_pixels is not None else sinogram.nb_bins_host )
        if self.nb_pixels < 1:
            raise ValueError( "nb_pixels doit être >= 1" )

        self.dw = sinogram.extent / self.nb_pixels
        self.s_min = sinogram.s_min
        # bords des cases image, [ nb_pixels + 1 ] -- côté backend (constante du graphe)
        self.edges = driver.array( self.s_min + self.dw * np.arange( self.nb_pixels + 1 ) )

        # borne (en éléments) du tenseur intermédiaire `[ nb_angles, nb_disques, nb_pixels + 1 ]` :
        # au-delà, `values` découpe la boucle sur les DISQUES en tranches et accumule -- le résultat
        # ne fait que `[ nb_angles, nb_pixels ]`, seul l'intermédiaire est gros.
        self.max_chunk_elems = int( max_chunk_elems )

        # la contribution d'une tranche, avec ses intermédiaires RECALCULÉS au backward au lieu
        # d'être gardés sur la bande (voir `values`). Enveloppé une fois ici, pas à chaque appel :
        # `driver.checkpoint` construit un objet de transformation, autant ne le faire qu'une fois.
        self._values_of_chunk = driver.checkpoint( self._values_of )

    # -- géométrie ---------------------------------------------------------

    @property
    def pixel_centers( self ) -> np.ndarray:
        """Centres des cases de la grille image, [ nb_pixels ]."""
        return self.s_min + self.dw * ( np.arange( self.nb_pixels ) + 0.5 )

    def _chunk_size( self, nb_disks: int ) -> int:
        per_disk = max( 1, self.nb_angles * ( self.nb_pixels + 1 ) )
        return max( 1, min( nb_disks, self.max_chunk_elems // per_disk ) )

    # -- projection --------------------------------------------------------

    def _values_of( self, centers, weights = None ):
        """Contribution (densité, `[ nb_angles, nb_pixels ]`) d'un paquet de centres, en tableau
        backend brut.

        `weights` (optionnel, `[ nb_disques ]`) pondère la contribution de chaque disque. Sert
        uniquement à NEUTRALISER les centres de remplissage d'une tranche incomplète (poids 0, voir
        `values`) : leur masse ET leur gradient deviennent exactement nuls. Pas de position « assez
        loin » qui ferait l'affaire à la place : quel que soit le point choisi, il existe un angle
        où sa projection retombe dans le détecteur.

        Le profil de Radon d'un disque de rayon r est la corde `c(u) = 2*sqrt(r^2 - u^2)` (nulle
        hors du disque), où `u = s - s0` est l'écart au centre projeté `s0 = centre.n_k`. On
        INTÈGRE cette corde sur chaque case (masse exactement conservée, comme `Sinogram.add_disk`)
        via sa primitive `H(u) = G(clip(u, -r, r))` avec
        `G(t) = t*sqrt(r^2 - t^2) + r^2*arcsin(t/r)`, puis on divise par la largeur de case pour
        obtenir une densité.

        DÉRIVÉE : `H` est C^1 et `H'(u) = c(u)` exactement -- mais autodifférentier l'expression
        de `G` ci-dessus est numériquement inutilisable. `G'` s'y présente comme une somme de deux
        termes en `1/sqrt(r^2 - t^2)` qui se compensent (annulation catastrophique quand un bord de
        case frôle le bord du disque), et là où `clip` sature la dérivée vaut `0 * inf = NaN`.
        On fournit donc la dérivée EXACTE à la main, par une surrogate du premier ordre :

            H = stop_gradient( G - c*u ) + stop_gradient( c ) * u

        dont la VALEUR est `G` (les `c*u` se compensent) et la DÉRIVÉE est `c(u)` -- la bonne, en
        une seule évaluation bien conditionnée. Aucun gradient ne traverse plus `sqrt`/`arcsin`/
        `clip`, donc plus de NaN ni d'annulation possible.
        """
        r = self.radius
        s0 = self.sinogram.project_points( centers ).tensor          # [ nb_angles, nb_disques ]
        u = self.edges[ None, None, : ] - s0[ :, :, None ]           # [ nb_angles, nb_disques, nb_pixels + 1 ]

        t = driver.clip( u, -r, r )
        sq = driver.sqrt( driver.clip( r * r - t * t, 0.0, None ) )
        chord = 2.0 * sq
        G = t * sq + r * r * driver.arcsin( driver.clip( t / r, -1.0, 1.0 ) )
        H = driver.stop_gradient( G - chord * u ) + driver.stop_gradient( chord ) * u

        mass = H[ :, :, 1: ] - H[ :, :, :-1 ]                        # masse par ( angle, disque, pixel )
        if weights is not None:
            mass = mass * weights[ None, :, None ]
        return driver.sum( mass, axis = 1 ) / self.dw                # somme sur les disques -> densité

    def values( self, centers ) -> Tensor:
        """Densité projetée `[ num_angle, num_pixel ]`, différentiable par rapport à `centers`
        (`[ nb_disques, 2 ]`, Tensor ou tableau).

        L'intermédiaire est de taille `nb_angles * nb_disques * ( nb_pixels + 1 )` -- bien plus gros
        que le résultat, `[ nb_angles, nb_pixels ]`. Au-delà de `max_chunk_elems` la somme sur les
        disques est donc découpée en TRANCHES accumulées une par une, ce qui ne change rien au
        résultat (à l'ordre de sommation près) et borne le pic mémoire à UNE tranche -- à condition
        de s'y prendre correctement, ce qui demande les deux mécanismes ensemble :

        - `driver.fold` (une boucle compilée, pas une boucle Python dépliée) rend l'exécution
          SÉQUENTIELLE. Dépliée, la boucle laisse au contraire le compilateur ordonnancer toutes les
          tranches de front -- mesuré : le pic restait proportionnel au nombre de disques, et le
          temps de compilation aussi ;
        - `driver.checkpoint` sur `_values_of` fait RECALCULER la tranche au backward au lieu d'en
          garder les intermédiaires, ce que la boucle stockerait sinon une fois par itération.

        Mesuré (600 angles x 2000 pixels, gradient) : 5.4 Go pour 512 disques avant, 0.4 Go après,
        et désormais indépendant du nombre de disques -- les ~5000 alvéoles de
        `experiments/lung_alveoli.py` demandaient plus de 50 Go, elles tiennent maintenant dans ce
        que prescrit `max_chunk_elems`. Le prix est un forward supplémentaire par tranche.

        La dernière tranche est complétée par des centres de remplissage neutralisés par un poids 0
        -- `driver.fold` exige des itérations de taille FIXE.
        """
        pts = centers if isinstance( centers, Tensor ) else RealTensor( centers )
        if pts.rank != 2 or pts.shape[ 1 ] != 2:
            raise ValueError( "centers doit être de shape [ nb_disques, 2 ]" )

        nb_disks = int( pts.shape[ 0 ] )
        chunk = self._chunk_size( nb_disks )
        raw = pts.tensor

        if chunk >= nb_disks:
            acc = self._values_of_chunk( raw )                       # une seule tranche : pas de boucle
        else:
            nb_chunks = -( -nb_disks // chunk )
            pad = nb_chunks * chunk - nb_disks
            # les centres de remplissage valent (0,0) : leur position n'a aucune importance
            # puisqu'ils portent un poids nul.
            padded = raw if pad == 0 else driver.pad( raw, ( ( 0, pad ), ( 0, 0 ) ) )
            weights = np.ones( nb_chunks * chunk )
            weights[ nb_disks: ] = 0.0

            xs = { "centers": padded.reshape( nb_chunks, chunk, 2 ),
                   "weights": driver.array( weights.reshape( nb_chunks, chunk ) ) }
            # la mise à jour du cumul reste HORS du checkpoint : dedans, le cumul lui-même
            # deviendrait un résidu stocké à chaque itération (voir `driver.fold`).
            acc = driver.fold(
                lambda cum, x: cum + self._values_of_chunk( x[ "centers" ], x[ "weights" ] ),
                driver.zeros( ( self.nb_angles, self.nb_pixels ) ), xs )

        return Tensor.wrap( acc, [ self.sinogram.num_angle.name, "num_pixel" ] )

    def image( self, centers ) -> Image:
        """La projection comme `Image` 1D batchée sur `num_angle` -- directement consommable comme
        distribution d'un `OtPlan1d`.

        `current_mass` est fourni explicitement (`somme des densités * largeur de case`, la
        définition même de la mesure d'une image constante par morceaux 1D) plutôt que laissé à
        `Image._update_current_mass`, qui la calcule par un `driver.call` (kernel C++). Le calcul
        est trivial en algèbre `Tensor` et l'éviter garde TOUTE la perte dans le graphe Jax : ni
        compilation C++ ni aller-retour FFI à chaque pas d'optimiseur.
        """
        values = self.values( centers )
        mass = values.sum( axis = "num_pixel" ) * self.dw            # [ num_angle ]
        return Image(
            values = values,
            origin = [ self.s_min ],                                 # partagé (géométrie commune)
            frame = [ [ self.dw ] ],                                 # partagé
            current_mass = mass,
            batch_axes = [ self.sinogram.num_angle ],
        )
