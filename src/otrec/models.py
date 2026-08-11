"""Les MODÈLES de reconstruction : ce qu'un nuage de points REPRÉSENTE, et ce qu'il en coûte.

Un `Model` répond à une seule question -- « quel est le coût de ce nuage de points, face au
sinogramme mesuré ? » -- par un scalaire différentiable par rapport aux points. C'est cette
fonction que `Reconstruction` (`Reconstruction.py`) fait décroître ; le modèle est le SEUL
endroit qui sait de quelle façon les points sont comparés à la donnée.

Les deux modèles disponibles s'appuient tous deux sur le transport optimal 1D semi-discret
(`OtPlan1d`, batché sur les angles), mais ils en ÉCHANGENT les rôles :

- `DiracModel` : les points sont l'INCONNUE vue comme des diracs de masse égale, projetés à la
  volée sur chaque détecteur ; la CIBLE est le profil mesuré, fonction constante par morceaux.
  Robuste et bon marché, mais la perte n'est pas différentiable partout (déplacement de diracs).
- `DiskModel` : les points sont les CENTRES de disques 2D de rayon FIXE ; c'est le sinogramme
  MESURÉ qui devient une somme de diracs pondérés (un par case détecteur, au centre de la case,
  de poids la valeur du pixel), et le MODÈLE est l'image constante par morceaux de la projection
  des disques (`disks.DiskProjector`). La perte y est lisse en les inconnues.

Ils exposent la même interface (`name`, `point_axis`, `cost`, `radii`, `floor`), si bien que
`Reconstruction` les enchaîne sans jamais tester leur type : un même nuage peut être convergé en
diracs puis raffiné en centres de disques.
"""
from abc import ABC, abstractmethod

from sdot import OtPlan1d, ProjectedSumOfDiracs, SumOfDiracs1d, Tensor

from .Sinogram import Sinogram
from .dirac_sycl import diracs_cost_grad
from .disks import DiskProjector


class Model( ABC ):
    """Interface commune des modèles. Un modèle est IMMUABLE et ne porte pas de points : il est
    construit une fois pour un sinogramme (+ ses paramètres) et évalué sur des nuages successifs.
    """

    #: nom lisible, pour les traces et l'historique de `Reconstruction`
    name = "model"

    #: nom de l'axe « point » des tenseurs de positions -- purement documentaire (il apparaît dans
    #: les shapes et les messages d'erreur), mais chaque modèle nomme ses points comme il les voit.
    point_axis = "num_point"

    #: rayon MONDE des points quand il fait PARTIE du modèle (`None` = points sans étendue propre).
    #: `Reconstruction.export_html` le transmet tel quel à `export_positions_html`, qui dessine
    #: alors les disques à leur vraie taille au lieu d'une taille d'affichage arbitraire.
    radii = None

    def __init__( self, sinogram: Sinogram ) -> None:
        self.sinogram = sinogram

    @abstractmethod
    def cost( self, points ) -> Tensor:
        """Coût scalaire (Tensor rang 0) du nuage `points` (`[ n, 2 ]`, Tensor ou tableau),
        différentiable par rapport à lui."""

    @property
    def floor( self ) -> float:
        """Coût INCOMPRESSIBLE : la valeur que `cost` atteint au mieux, même à la vérité terrain.
        Sert de référence pour juger une reconstruction (0 quand il n'y en a pas)."""
        return 0.0

    def wrap( self, raw ) -> Tensor:
        """Le tableau backend `raw` (`[ n, 2 ]`) présenté comme le Tensor de points du modèle."""
        return Tensor.wrap( raw, [ self.point_axis, "dim" ] )

    def __repr__( self ) -> str:
        return f"{ type( self ).__name__ }( { self.name } )"


class DiracModel( Model ):
    """Les points sont des DIRACS de masse égale : la densité reconstruite est leur somme.

    La projection `s = point.n_k` N'EST PAS matérialisée : `ProjectedSumOfDiracs` garde les points
    2D PARTAGÉS (une seule copie pour tous les angles) et la normale PAR ANGLE, le kernel calculant
    la position 1D à la volée -- au lieu d'un tenseur `[ nb_angles, n ]` (80 Go à 1e7 diracs x 1000
    angles). Reste différentiable : le backward scatter-atomique le gradient de la position
    projetée sur les points 2D partagés.

    `with_barycenters` : transmis tel quel à `OtPlan1d`. Quand le SEUL gradient demandé est celui
    des positions (le cas ici), stocker les barycentres évite au backward de re-trier + re-balayer
    chaque angle, au prix d'un buffer `[ nb_angles, n ]` -- à activer si ce coût mémoire est
    acceptable (voir la docstring d'`OtPlan1d.__init__`).
    """

    name = "diracs"
    point_axis = "num_dirac"

    def __init__( self, sinogram: Sinogram, with_barycenters: bool = False ) -> None:
        super().__init__( sinogram )
        self.with_barycenters = with_barycenters

    def cost( self, points ) -> Tensor:
        pts = points if isinstance( points, Tensor ) else Tensor( points )
        src = ProjectedSumOfDiracs( points = pts, normal = self.sinogram.normals_t,
                                    batch_axes = [ self.sinogram.num_angle ] )
        dst = self.sinogram.batched_image()
        return OtPlan1d( src, dst, with_barycenters = self.with_barycenters ).cost.sum()  # somme sur les angles

    def value_and_grad( self, points ):
        """`(cost, grad)` fusionnés via le kernel SYCL (`dirac_sycl.diracs_cost_grad`) -- même
        formule que `cost` + `jax.grad`, mais calculée EN UN SEUL passage (voir sa docstring), sans
        passer par l'autodiff Jax. `points` : `[ n, proj_dim ]`, Tensor ou tableau brut -- pas
        besoin de `wrap` (contrairement à `cost`). `with_barycenters` n'a pas de sens ici (pas de
        bwd Jax) : ignoré. Consommé par `optimizers.FusedLBFGS` (voir
        `Reconstruction.diracs( backend = "sycl" )`)."""
        return diracs_cost_grad( points, self.sinogram )


class DiskModel( Model ):
    """Les points sont les CENTRES de disques 2D de rayon FIXE.

    Le sinogramme mesuré est ici la source discrète (`sinogram_diracs`) et la projection des
    disques (`disks.DiskProjector`) la cible continue. Chaque tranche (angle) normalise ses deux
    distributions à la masse 1 : ni le nombre de disques ni le rayon n'ont besoin d'être calibrés
    sur la masse mesurée, seule compte la FORME du profil.

    `nb_pixels` : finesse de la grille du modèle, sur la même étendue que le détecteur (par défaut
    celle du détecteur). La raffiner représente mieux la projection d'un petit rayon sans toucher
    aux données mesurées -- c'est un choix LIBRE, indépendant de `nb_bins`.

    `max_chunk_elems` : la taille des tranches de disques traitées une par une, donc le PIC MÉMOIRE
    du gradient -- borné à une tranche quel que soit le nombre de disques (voir
    `DiskProjector.values`). Mesuré sur le cas du poumon (600 angles x 2000 pixels), le temps ne
    varie que de ~20% entre 3 et 223 disques par tranche : le calcul est limité par la bande
    passante mémoire, pas par le nombre de tranches -- inutile donc de monter cette borne pour
    aller plus vite, elle ne sert qu'à choisir la mémoire qu'on accepte de consommer.
    """

    name = "disques"
    point_axis = "num_disk"

    def __init__( self, sinogram: Sinogram, radius: float, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24 ) -> None:
        super().__init__( sinogram )
        self.projector = DiskProjector( sinogram, radius = radius, nb_pixels = nb_pixels,
                                        max_chunk_elems = max_chunk_elems )
        self.radii = self.projector.radius       # rayon commun, exporté tel quel à la visualisation

    def cost( self, points ) -> Tensor:
        src = sinogram_diracs( self.sinogram )
        dst = self.projector.image( points )
        return OtPlan1d( src, dst ).cost.sum()                        # somme sur les angles

    @property
    def floor( self ) -> float:
        """Plancher de QUANTIFICATION : même à centres exacts la perte ne vaut pas 0, car les
        diracs condensent chaque case détecteur en son centre -- ce qui coûte la variance d'une
        case uniforme, `dw^2 / 12`, par angle."""
        return float( self.sinogram.nb_angles.value ) * self.sinogram.dw ** 2 / 12


def sinogram_diracs( sinogram: Sinogram ) -> SumOfDiracs1d:
    """Le sinogramme MESURÉ vu comme une somme de diracs pondérés, batchée sur les angles : un
    dirac par case détecteur, placé au CENTRE de la case et de poids la valeur du pixel.

    Les positions (centres de cases) sont les mêmes à tous les angles : elles sont donc PARTAGÉES
    (`[ nb_bins ]`, pas `[ nb_angles, nb_bins ]`) ; seuls les poids sont batchés. C'est ce que
    `raw_1d_diracs` transmet ensuite au chemin pur Jax d'`Image.try_update_otplan1d`, qui ne
    matérialise jamais plus d'un angle à la fois.
    """
    return SumOfDiracs1d(
        positions = sinogram.bin_centers,
        weights = sinogram.values,
        batch_axes = [ sinogram.num_angle ],
    )
