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

En 3D ( des `Radiographs`, une image 2D par angle ), `ProjectedDiracModel` joue le rôle de
`DiracModel` avec un transport semi-discret 2D par angle ( `OtPlan` ) -- et ne propose que
l'évaluation FUSIONNÉE coût + gradient ( voir sa docstring ).
"""
import warnings
from abc import ABC, abstractmethod

from loom import Tensor
from loom import RealTensor
import numpy as np

from sdot import OtPlan, OtPlan1d, ProjectedSumOfDiracs, SumOfDiracs, SumOfDiracs1d

from .Radiographs import Radiographs
from .Sinogram import Sinogram
from .dirac_sycl import diracs_cost, diracs_cost_grad
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
        pts = points if isinstance( points, Tensor ) else RealTensor( points )
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

    def value( self, points ) -> float:
        """`cost` SEUL (flottant Python), MÊME kernel SYCL fusionné que `value_and_grad` mais SANS
        le calcul de gradient (`dirac_sycl.diracs_cost`) -- pour les évaluations "coût seul" d'une
        recherche de pas, où le gradient serait de toute façon jeté."""
        return diracs_cost( points, self.sinogram )


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


class ProjectedDiracModel( Model ):
    """Les points sont des DIRACS 3D de masse égale, confrontés à des RADIOGRAPHIES
    ( `Radiographs` ) : à chaque angle, leurs projections sur le détecteur sont transportées vers
    l'image mesurée par un transport semi-discret 2D ( `OtPlan`, un diagramme de puissance par
    angle ), et le coût est la somme sur les angles des `W_2^2`.

    Le pendant 3D de `DiracModel`, avec une différence de nature : le transport 1D est EXACT ( un
    tri ), le transport 2D est un AJUSTEMENT de poids ( `OtPlan._fit`, itératif ). D'où :

    - le gradient par rapport aux points ne passe pas par l'autodiff mais par le théorème de
      l'ENVELOPPE, aux poids ajustés : `2 m_i ( p_i - b_i )` sur le détecteur ( `b_i` le barycentre
      de la cellule de Laguerre, voir `OtPlan.cost_and_position_grad` ), puis remonté en 3D par la
      transposée de la projection ( `Radiographs.unproject_grad` ). Le modèle n'a donc qu'un
      `value_and_grad` fusionné ( `FusedLBFGS` ) -- `cost` rend un flottant, pas un `Tensor` ;
    - chaque angle est résolu par le Newton amorti de `OtPlan` ( tout en C++ ), et les
      poids ajustés sont GARDÉS d'une évaluation à l'autre ( `weights0` du prochain `OtPlan` ) :
      des points qui bougent peu demandent des poids qui bougent peu, quelques pas suffisent. C'est
      un cache, pas un état -- le résultat n'en dépend pas, et il n'a plus à être écarté quand il
      vide une cellule : le C++ choisit lui-même le meilleur des trois départs qu'il connaît ( les
      poids donnés, le Voronoï, la similitude qui ramène le nuage dans le détecteur ) et dit lequel
      dans `stats[ "depart" ]` ;
    - une projection est 2D, donc le pas de Newton y est celui par les LIMITES ( `step = "auto"`
      -> `"limits"` ) : le coefficient de relaxation maximal des cellules qui s'écrasent le long de
      la direction, calculé EXACTEMENT au lieu d'être cherché par essais successifs -- d'où des
      reculs devenus rares ( 1.8 par ajustement sur le cas mesuré ci-dessous, contre 515 quand le
      problème est mal posé ) ;
    - la radiographie reçoit un FOND ( `background`, en fraction de sa valeur moyenne ), et il
      reste INDISPENSABLE : une boule projetée est nulle hors de son ombre, et une cellule qui ne
      voit que des zéros n'a AUCUN poids qui lui donne sa masse -- le problème n'a pas de solution,
      pas seulement un mauvais départ. La CONTINUATION EN LARGEUR d'`OtPlan`
      ( `continuation = "auto"` : la densité convolée large d'abord, resserrée étape par étape,
      chacune repartant des poids de la précédente ) adoucit le CHEMIN, pas la cible : sa dernière
      étape est la densité elle-même, zéros compris. Mesuré ( 200 diracs, 6 angles, 64 x 64,
      départ dans le cube, `notes/2026-09-23-otrec-3d.md` ) : sans fond, 36 ajustements sur 48 ne convergent
      pas, 515 reculs par appel, et la perte MONTE ( 2.6 -> 8.0 ) ; avec un fond de 1e-6, tous
      convergent, 22 diagrammes par appel, 98 % des diracs finissent dans les boules ;
    - ce que la continuation change, en revanche, c'est qu'un fond FAIBLE devient utilisable.
      Même cas, `continuation = "never"` contre `"auto"`, à fond 1e-6 : 25 ajustements sur 78 ne
      convergent pas, 950 diagrammes par appel, 46 % des diracs dans les boules -- contre aucun
      raté, 22 diagrammes et 98 %. À fond 1e-3 elle ne coûte rien et économise un tiers des
      diagrammes en supprimant les reculs ;
    - le point de DÉPART compte toujours, pour ce qu'il coûte : des diracs tirés dans tout le cube
      font refaire des étapes de continuation à chaque angle et à chaque évaluation, là où
      `Reconstruction.hull_points` ( l'enveloppe visuelle ) part d'emblée près de l'objet.
    """

    name = "diracs 3D"
    point_axis = "num_dirac"
    #: pas de `cost` traçable : seul `value_and_grad` existe ( voir la docstring )
    fused_only = True

    def __init__( self, radiographs: Radiographs, background: float = 1e-3, max_iter: int = 100,
                  mass_tol: float = 1e-4, kernel_dtype = None, continuation: str = "auto",
                  strict: bool = False ) -> None:
        """`background`, `max_iter`, `mass_tol`, `continuation` : voir la docstring de la classe et
        `OtPlan`. Le Newton est celui de KMT, amorti ( le Newton NON amorti a été mesuré 5 à 50 fois
        plus lent, floutage ou pas -- voir `notes/2026-09-14-reconstruction-3d.md` -- et n'existe
        plus ), le pas venant des LIMITES puisqu'une projection est 2D.

        `mass_tol` est RELATIF à la masse d'un dirac ( `1 / n` ) -- et borné par ce que le noyau
        sait : en FP32, l'aire d'une cellule n'est connue qu'à ~1e-5 près en relatif, en dessous le
        Newton ne trouve plus de pas qui baisse le résidu. D'où `kernel_dtype = None` par défaut,
        qui laisse `OtPlan` couper en FP64 -- FP32 est à réserver aux essais.

        `strict` : lever dès qu'un angle ne converge pas, au lieu de le compter dans
        `solver_stats` et de rendre quand même le coût ( un ajustement inachevé donne un gradient
        faux par le théorème de l'enveloppe, qui suppose les poids optimaux )."""
        super().__init__( radiographs )
        self.radiographs = radiographs
        self.background = float( background )
        self.max_iter = int( max_iter )
        self.mass_tol = float( mass_tol )
        self.kernel_dtype = kernel_dtype
        self.continuation = continuation
        self.strict = bool( strict )
        nb_angles = int( radiographs.nb_angles.value )
        self._images = [ radiographs.image( k, background = self.background ) for k in range( nb_angles ) ]
        self._weights = [ None ] * nb_angles
        # une cible qui garde des pixels NULS ne se transporte pas ( voir la docstring de la classe :
        # mesuré, les ajustements n'y convergent pas et la perte monte ). On le dit une fois, au lieu
        # de laisser une descente de plusieurs heures rendre un nuage faux.
        if self.background <= 0 and float( np.asarray( radiographs.values ).min() ) <= 0:
            warnings.warn( "ProjectedDiracModel : les radiographies ont des pixels NULS et aucun fond "
                           "n'est ajouté ( background = 0 ) -- les ajustements par angle ne convergeront "
                           "pas ( la continuation adoucit le chemin, pas la cible ). Donner "
                           "`background > 0` ( 1e-6 suffit ).", stacklevel = 2 )
        #: ce que les ajustements ont coûté depuis le début, tous angles confondus ( voir
        #: `solver_line` ) -- ce qu'on regarde pour savoir si le nuage est encore loin
        self.solver_stats = dict( nb_calls = 0, nb_iter = 0, nb_diag = 0, nb_recul = 0, nb_etapes = 0,
                                  nb_voronoi = 0, nb_similitude = 0, nb_not_converged = 0 )

    def _plan( self, k, uv ):
        """le transport de l'angle `k`, ajusté -- en repartant des poids de la dernière fois"""
        # le NEWTON sur la fonctionnelle duale : un nombre de pas indépendant du nombre de diracs,
        # et, d'une évaluation à l'autre ( `weights0` ), quelques pas seulement. Des poids qui
        # vident une cellule ne sont plus écartés ici : le C++ compare lui-même les trois départs
        # qu'il connaît et garde le meilleur ( `stats[ "depart" ]` ).
        kw = dict( max_iter = self.max_iter, mass_tol = self.mass_tol / len( uv ),
                   kernel_dtype = self.kernel_dtype, continuation = self.continuation )
        # un nuage qui a changé de TAILLE ( un étage de `Reconstruction.multiscale` ) repart de zéro
        w0 = self._weights[ k ]
        if w0 is not None and len( w0 ) != len( uv ):
            w0 = None
        plan = OtPlan( SumOfDiracs( uv ), self._images[ k ], weights0 = w0, **kw )
        self._weights[ k ] = plan.weights
        self._account( k, plan )
        return plan

    def _account( self, k, plan ):
        """ce que l'ajustement de l'angle `k` a coûté, cumulé dans `solver_stats`"""
        st = plan.stats
        s = self.solver_stats
        s[ "nb_calls" ] += 1
        for name in ( "nb_iter", "nb_diag", "nb_recul", "nb_etapes" ):
            s[ name ] += st[ name ]
        if st[ "depart" ] == "voronoi":
            s[ "nb_voronoi" ] += 1
        elif st[ "depart" ] == "similitude":
            s[ "nb_similitude" ] += 1
        if not plan.converged:
            s[ "nb_not_converged" ] += 1
            if self.strict:
                raise RuntimeError( f"ProjectedDiracModel : l'angle { k } n'a pas convergé "
                                    f"( { st[ 'fin' ] }, reste { st[ 'reste' ]:.3e} ) -- le gradient de "
                                    "l'enveloppe suppose les poids optimaux" )

    def solver_line( self ) -> str:
        """Une ligne de ce que les ajustements ont coûté, MOYENNÉE par angle résolu -- de quoi voir
        d'un coup d'oeil si le nuage est encore loin ( des étapes de continuation, des départs au
        Voronoï ) ou déjà chaud ( deux ou trois pas de Newton, aucune étape )."""
        s = self.solver_stats
        nb = max( 1, s[ "nb_calls" ] )
        return ( f"{ s[ 'nb_calls' ] } ajustements : { s[ 'nb_iter' ] / nb:.1f} pas, "
                 f"{ s[ 'nb_diag' ] / nb:.1f} diagrammes, { s[ 'nb_etapes' ] / nb:.2f} étapes de continuation, "
                 f"{ s[ 'nb_recul' ] / nb:.2f} reculs, départs { s[ 'nb_voronoi' ] } Voronoï / "
                 f"{ s[ 'nb_similitude' ] } similitude, { s[ 'nb_not_converged' ] } non convergés" )

    def value_and_grad( self, points ):
        """`( cost, grad )`, `grad` de shape `[ n, 3 ]` -- voir la docstring de la classe."""
        pts = np.asarray( points, dtype = float ).reshape( -1, 3 )
        proj = self.radiographs.project_points( pts )                           # [ nb_angles, n, 2 ]
        cost, grad_uv = 0.0, np.zeros_like( proj )
        for k in range( len( proj ) ):
            c, g = self._plan( k, proj[ k ] ).cost_and_position_grad()
            cost += c
            grad_uv[ k ] = g
        return cost, self.radiographs.unproject_grad( grad_uv )

    def value( self, points ) -> float:
        pts = np.asarray( points, dtype = float ).reshape( -1, 3 )
        proj = self.radiographs.project_points( pts )
        return float( sum( self._plan( k, proj[ k ] ).cost for k in range( len( proj ) ) ) )

    def cost( self, points ):
        """Le coût -- un FLOTTANT ( pas dérivable par autodiff, voir la docstring )."""
        return self.value( points )

    def wrap( self, raw ) -> Tensor:
        return Tensor.wrap( raw, [ self.point_axis, "dim" ] )


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
