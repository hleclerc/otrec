"""Générateur de sinogrammes 2D synthétiques.

Un `Sinogram` représente la donnée MESURÉE d'un problème de reconstruction : les
projections 1D (transformée de Radon) d'un objet 2D, échantillonnées sur un
détecteur discrétisé, pour un jeu d'angles.

Usage : on part de zéro et on accumule des primitives dont la projection est
connue analytiquement (`add_disk` pour le 2D). Les valeurs vivent dans le champ
tenseur `values[ num_angle, num_bin ]`.

Conventions géométriques :
- angles θ_k = k·π/nb_angles, régulièrement répartis sur [0, π) ;
- normale de projection n_θ = (cos θ, sin θ) ; la coordonnée détecteur d'un point
  p est s = p·n_θ (projection le long de la direction perpendiculaire à n_θ) ;
- le détecteur couvre [ center − extent/2, center + extent/2 ], découpé en
  `nb_bins` cases de largeur dw = extent/nb_bins.

`values[ k ]` est la fonction constante par morceaux du profil à l'angle k ;
`image( k )` la présente comme `Image` 1D (en coordonnées détecteur réelles),
directement consommable comme distribution cible d'un `OtPlan1d`.
"""
import numpy as np

from loom import ShapeVar, Axis, Tensor, Aggregate, RealTensor
from sdot import Image


class Sinogram( Aggregate ):
    nb_angles : ShapeVar
    nb_bins   : ShapeVar

    num_angle : Axis[ "nb_angles" ]
    num_bin   : Axis[ "nb_bins" ]

    values    : RealTensor[ "num_angle", "num_bin" ]

    def __init__( self, nb_angles: int, nb_bins: int, extent: float, detector_center: float = 0.0 ) -> None:
        if nb_angles < 1 or nb_bins < 1:
            raise ValueError( "nb_angles et nb_bins doivent être >= 1" )
        if extent <= 0:
            raise ValueError( "extent doit être > 0" )

        # géométrie détecteur / angulaire (attributs hôtes, non tenseurs : la génération
        # de données n'est pas différentiée et se calcule le plus simplement en numpy)
        self.extent = float( extent )
        self.detector_center = float( detector_center )
        # le compte de cases, côté HÔTE. Doublon apparent de la ShapeVar `nb_bins`, mais il est
        # disponible ICI, avant que `values` ne soit posé : `nb_bins.value` se résout depuis les
        # tailles observées, donc pas avant `__base_init__` plus bas -- alors que `dw`/`s_min` en
        # ont besoin tout de suite. La géométrie détecteur est de la donnée HÔTE (comme
        # `angles`/`normals`) et doit le rester : elle est constante pendant toute une optimisation.
        # (`nb_bins.value` est désormais un `ShapeArray`, donc un compte hôte lui aussi -- ce n'est
        # plus le `Tensor` traçable qui rendait ce doublon obligatoire.)
        self.nb_bins_host = int( nb_bins )
        self.dw = self.extent / nb_bins
        self.s_min = self.detector_center - self.extent / 2

        angles = np.pi * np.arange( nb_angles ) / nb_angles                        # [ nb_angles ]
        self.angles = angles
        self.normals = np.stack( [ np.cos( angles ), np.sin( angles ) ], axis = 1 )  # [ nb_angles, 2 ]
        # même géométrie, côté Tensor : sert la projection différentiable (`project_points`). Les
        # normales portent un axe coordonnée `_xy` (dim 2) PARTAGÉ, sur lequel la projection contracte
        # PAR RÉFÉRENCE (pas de `@` qui supposerait un ordre d'axes). L'axe des angles leur est propre.
        self._xy = Axis( ShapeVar( 2 ) )
        self.normals_t = RealTensor[ Axis( ShapeVar( int( nb_angles ) ) ), self._xy ]( self.normals )

        # les valeurs démarrent à 0 ; `add_disk` les accumule
        self.__base_init__(
            values = np.zeros( ( int( nb_angles ), int( nb_bins ) ), dtype = float ),
        )

    # -- géométrie ---------------------------------------------------------

    @property
    def bin_edges( self ) -> np.ndarray:
        """Bords des cases détecteur, [ nb_bins + 1 ]."""
        return self.s_min + self.dw * np.arange( self.nb_bins_host + 1 )

    @property
    def bin_centers( self ) -> np.ndarray:
        """Centres des cases détecteur, [ nb_bins ]."""
        return self.s_min + self.dw * ( np.arange( self.nb_bins_host ) + 0.5 )

    def project_points( self, points ) -> Tensor:
        """Coordonnées détecteur des `points` (shape [ n, 2 ]) pour chaque angle.

        Renvoie un `Tensor` [ nb_angles, n ] : s[ k, i ] = points[ i ] · n_θk. La projection est une
        CONTRACTION PAR RÉFÉRENCE sur l'axe coordonnée partagé `_xy` (`normals.dot( points, over=_xy )`)
        -- aucun `@`, donc aucune hypothèse d'ordre d'axes. Tout passe par l'algèbre `Tensor`, donc
        c'est DIFFÉRENTIABLE et compatible trace (ce qu'exige le gradient de la reconstruction).
        """
        val = points if isinstance( points, Tensor ) else RealTensor( points )
        if val.rank != 2 or val.shape[ 1 ] != 2:
            raise ValueError( "points doit être de shape [ n, 2 ]" )
        # les points partagent l'axe `_xy` des normales ; leur axe « point » leur est propre
        pts = RealTensor[ Axis( ShapeVar() ), self._xy ]( val )
        return self.normals_t.dot( pts, over = self._xy )                          # [ nb_angles, n ]

    # -- accumulation ------------------------------------------------------

    def add_disk( self, center, radius: float, density: float = 1.0 ) -> "Sinogram":
        """Ajoute la projection d'un disque uniforme (2D).

        Le profil de Radon d'un disque de rayon r et densité ρ est la corde
        `ρ·2·√(r² − t²)` où t = s − s0 est l'écart au centre projeté
        s0 = center·n_θ. On INTÈGRE ce profil sur chaque case pour préserver la
        masse : la valeur stockée est la densité moyenne sur la case
        (mass = value·dw), si bien que la masse totale par angle vaut
        exactement ρ·π·r² tant que le disque tient dans le détecteur.

        Retourne self pour permettre le chaînage.
        """
        center = np.asarray( center, dtype = float )
        if center.shape != ( 2, ):
            raise ValueError( "center doit être de shape [ 2 ]" )
        if radius <= 0:
            raise ValueError( "radius doit être > 0" )

        r = float( radius )
        s0 = self.normals @ center                                                 # [ nb_angles ]

        # écart des bords de cases au centre projeté, borné à [ −r, r ]
        edges = self.bin_edges[ None, : ] - s0[ :, None ]                          # [ nb_angles, nb_bins + 1 ]
        t = np.clip( edges, -r, r )

        # primitive de 2·√(r² − t²) : G(t) = t·√(r² − t²) + r²·arcsin( t / r )
        G = t * np.sqrt( np.maximum( r * r - t * t, 0.0 ) ) + r * r * np.arcsin( t / r )
        contribution = density * ( G[ :, 1: ] - G[ :, :-1 ] ) / self.dw            # [ nb_angles, nb_bins ]

        self.values = self.values + contribution
        return self

    # -- consommation ------------------------------------------------------

    def image( self, k: int ) -> Image:
        """`Image` 1D du profil à l'angle k, en coordonnées détecteur réelles.

        Utilisable directement comme distribution cible d'un `OtPlan1d`.
        """
        return Image(
            values = self.values[ k ],
            origin = [ self.s_min ],
            frame = [ [ self.dw ] ],
        )

    def batched_image( self ) -> Image:
        """Tous les profils d'un coup : une `Image` BATCHÉE sur `num_angle` (l'axe de batch existe
        déjà ici -- c'est celui des `values`). `origin`/`frame`, identiques à tous les angles, sont
        PARTAGÉS (non batchés). Sert de distribution cible à un `OtPlan1d` batché, sans boucle Python.
        """
        return Image(
            values = self.values,                       # [ num_angle, num_bin ]
            origin = [ self.s_min ],                    # partagé (géométrie détecteur commune)
            frame = [ [ self.dw ] ],                    # partagé
            batch_axes = [ self.num_angle ],
        )

    def mass( self, k: int | None = None ):
        """Masse totale (∫ profil ds) à l'angle k, ou par angle (rang 1) si k est None."""
        m = self.values.sum( axis = "num_bin" ) * self.dw
        return m if k is None else m[ k ]

    def debias_and_equalize_mass( self ) -> "Sinogram":
        """Corrige un sinogramme dont l'objet DÉPASSE le détecteur (rayon > extent/2) : la
        fenêtre visible ne redescend alors plus jusqu'à 0 (on ne voit qu'un morceau central de
        l'objet), et la masse mesurée VARIE selon l'angle (la largeur d'ombre tronquée dépend de
        l'orientation). `OtPlan1d` rétablit déjà l'égalité de masse src/dst via
        `normalized_version` (un RESCALE multiplicatif par angle) -- correct pour une vraie
        distribution de densité, mais ici ça revient à gonfler artificiellement la partie visible
        d'un angle très tronqué, déformant la forme reconstruite. Un décalage ADDITIF est plus
        fidèle : on modélise la partie invisible comme un fond localement homogène par angle
        (raisonnable si l'objet est beaucoup plus grand que la fenêtre), qu'on retire avant toute
        reconstruction.

        Cherche une constante `c_k` par angle telle que :
          - la masse résultante `mass_k - c_k * extent` soit la MÊME pour tous les angles (M),
          - le minimum GLOBAL (tous angles, tous bins confondus) du sinogramme corrigé soit
            exactement 0 -- pas nécessairement le minimum de CHAQUE angle individuellement :
            l'angle le plus tronqué (celui qui autoriserait la plus petite masse en retirant tout
            son propre fond) fixe M, les autres angles retirent une constante plus petite que leur
            propre minimum pour rejoindre cette même masse M.

        Dérivation (voir docstring de la classe pour les notations `mass_k`/`m_k` = min du profil
        brut de l'angle k) : en écrivant `min_k( m_k − c_k ) = 0` et `c_k = ( mass_k − M ) /
        extent`, on obtient `M = max_k( mass_k − m_k·extent )`.
        """
        vals = np.asarray( self.values )
        m = vals.min( axis = 1 )                                  # [ nb_angles ]
        mass = vals.sum( axis = 1 ) * self.dw                     # [ nb_angles ]
        M = float( np.max( mass - m * self.extent ) )
        c = ( mass - M ) / self.extent                            # [ nb_angles ], <= m partout

        out = Sinogram( nb_angles = self.nb_angles.value, nb_bins = self.nb_bins.value,
                         extent = self.extent, detector_center = self.detector_center )
        out.values = vals - c[ :, None ]
        return out
