"""Générateur de RADIOGRAPHIES synthétiques : le pendant 3D de `Sinogram`.

Un `Radiographs` représente la donnée MESURÉE d'une reconstruction 3D : pour chaque angle, la
projection 2D ( intégrale le long des rayons, faisceau parallèle ) d'un objet 3D, échantillonnée
sur un détecteur plan discrétisé. Là où le sinogramme 2D empile des PROFILS 1D, on empile ici des
IMAGES 2D : `values[ num_angle, num_u, num_v ]`.

Usage : on part de zéro et on accumule des primitives dont la projection est connue analytiquement
( `add_sphere` ). `image( k )` / `batched_image()` présentent les radiographies comme des `Image`
2D, consommables comme distribution cible d'un `OtPlan` ( le transport semi-discret 2D ).

Conventions géométriques -- une rotation autour de l'axe `z`, comme un tomographe :
- angles θ_k = k·π/nb_angles, régulièrement répartis sur [0, π) ;
- direction de projection ( le rayon ) d_θ = ( cos θ, sin θ, 0 ) ;
- axes du détecteur u_θ = ( −sin θ, cos θ, 0 ) et v = ( 0, 0, 1 ) ; les coordonnées détecteur
  d'un point p sont ( u, v ) = ( p·u_θ, p·v ). À θ = 0, `u = y` et `v = z` -- et `Sinogram`, à
  θ = 0, mesure `s = x` : les deux conventions diffèrent d'un quart de tour, ce qui n'a aucune
  conséquence ( un angle est un angle ) mais mérite d'être dit ;
- le détecteur couvre [ center_u − extent_u/2, center_u + extent_u/2 ] × [ idem en v ], découpé
  en `nb_u × nb_v` pixels de côtés `du`, `dv`.
"""
import numpy as np

from loom import ShapeVar, Axis, Aggregate, RealTensor
from sdot import Image


class Radiographs( Aggregate ):
    nb_angles : ShapeVar
    nb_u      : ShapeVar
    nb_v      : ShapeVar

    num_angle : Axis[ "nb_angles" ]
    num_u     : Axis[ "nb_u" ]
    num_v     : Axis[ "nb_v" ]

    values    : RealTensor[ "num_angle", "num_u", "num_v" ]

    #: la dimension de l'espace où vivent les objets ( ce que `Reconstruction` lit pour tirer des
    #: points de la bonne taille ) -- `Sinogram` vaut 2
    world_dim = 3

    def __init__( self, nb_angles: int, nb_u: int, nb_v: int, extent_u: float, extent_v: float | None = None,
                  detector_center = ( 0.0, 0.0 ), quadrature: int = 4 ) -> None:
        """`nb_u × nb_v` pixels sur `extent_u × extent_v` ( `extent_v = extent_u` par défaut ).

        `quadrature` : le nombre de points de Gauss-Legendre PAR AXE avec lesquels `add_sphere`
        intègre la projection sur chaque pixel ( voir sa docstring ).
        """
        if nb_angles < 1 or nb_u < 1 or nb_v < 1:
            raise ValueError( "nb_angles, nb_u et nb_v doivent être >= 1" )
        extent_v = extent_u if extent_v is None else extent_v
        if extent_u <= 0 or extent_v <= 0:
            raise ValueError( "extent_u et extent_v doivent être > 0" )

        # géométrie détecteur / angulaire : de la donnée HÔTE, constante pendant toute une
        # reconstruction ( voir `Sinogram` pour la même remarque )
        self.extent_u, self.extent_v = float( extent_u ), float( extent_v )
        #: le côté du plus grand cube centré dont TOUTE projection tient dans le détecteur -- ce
        #: dans quoi `Reconstruction.random_points` tire ( la diagonale d'une face tourne en `u` )
        self.extent = min( self.extent_u / np.sqrt( 2 ), self.extent_v )
        self.detector_center = ( float( detector_center[ 0 ] ), float( detector_center[ 1 ] ) )
        self.nb_u_host, self.nb_v_host = int( nb_u ), int( nb_v )
        self.du = self.extent_u / nb_u
        self.dv = self.extent_v / nb_v
        self.u_min = self.detector_center[ 0 ] - self.extent_u / 2
        self.v_min = self.detector_center[ 1 ] - self.extent_v / 2
        self.quadrature = int( quadrature )

        angles = np.pi * np.arange( nb_angles ) / nb_angles                        # [ nb_angles ]
        self.angles = angles
        c, s = np.cos( angles ), np.sin( angles )
        z = np.zeros_like( angles )
        self.directions = np.stack( [ c, s, z ], axis = 1 )                      # [ nb_angles, 3 ]
        # la base du détecteur, `[ nb_angles, 2, 3 ]` : la ligne 0 est `u_θ`, la ligne 1 est `v`
        self.bases = np.stack( [ np.stack( [ -s, c, z ], axis = 1 ),
                                 np.stack( [ z, z, np.ones_like( angles ) ], axis = 1 ) ], axis = 1 )

        self.__base_init__(
            values = np.zeros( ( int( nb_angles ), int( nb_u ), int( nb_v ) ), dtype = float ),
        )

    # -- géométrie ---------------------------------------------------------

    @property
    def u_edges( self ) -> np.ndarray:
        return self.u_min + self.du * np.arange( self.nb_u_host + 1 )

    @property
    def v_edges( self ) -> np.ndarray:
        return self.v_min + self.dv * np.arange( self.nb_v_host + 1 )

    @property
    def u_centers( self ) -> np.ndarray:
        return self.u_min + self.du * ( np.arange( self.nb_u_host ) + 0.5 )

    @property
    def v_centers( self ) -> np.ndarray:
        return self.v_min + self.dv * ( np.arange( self.nb_v_host ) + 0.5 )

    def project_points( self, points ) -> np.ndarray:
        """Coordonnées détecteur `[ nb_angles, n, 2 ]` des `points` ( `[ n, 3 ]` ), pour chaque angle.

        Côté HÔTE, en numpy : la reconstruction 3D dérive son coût par le théorème de l'enveloppe
        ( `models.ProjectedDiracModel` ), pas par autodiff, donc la projection n'a pas besoin d'être
        tracée. Sa transposée est `unproject_grad`.
        """
        pts = np.asarray( points, dtype = float )
        if pts.ndim != 2 or pts.shape[ 1 ] != 3:
            raise ValueError( "points doit être de shape [ n, 3 ]" )
        return np.einsum( "kcd,nd->knc", self.bases, pts )

    def unproject_grad( self, grad_uv ) -> np.ndarray:
        """L'adjoint de `project_points` : des cotangentes `[ nb_angles, n, 2 ]` sur les
        coordonnées détecteur, la cotangente `[ n, 3 ]` sur les points -- la somme sur les angles
        de `g_k . base_k`."""
        return np.einsum( "knc,kcd->nd", np.asarray( grad_uv, dtype = float ), self.bases )

    def visual_hull_points( self, nb_points: int, seed: int = 0, extent: float | None = None,
                            threshold: float = 0.0 ) -> np.ndarray:
        """`nb_points` points tirés uniformément dans l'ENVELOPPE VISUELLE : le cube
        `[ -extent/2, extent/2 ]^3` ( `self.extent` par défaut ) réduit aux points dont TOUTES les
        projections tombent sur un pixel de valeur `> threshold` -- par rejet.

        Le point de départ qu'une reconstruction veut : un dirac dont une projection tombe dans le
        vide n'a, à cet angle, qu'une cellule de mesure quasi nulle, et le transport qui doit
        l'amener jusqu'à l'ombre est aussi mal conditionné qu'il est loin ( voir `OtPlan`,
        `objective = "newton"` ). L'enveloppe visuelle contient l'objet, et c'est déjà lui à peu
        de choses près quand les angles sont assez nombreux.
        """
        rng = np.random.default_rng( seed )
        e = float( self.extent if extent is None else extent )
        vals = np.asarray( self.values )
        nb_angles = len( self.angles )
        pts, tried = [], 0
        while sum( len( p ) for p in pts ) < nb_points:
            batch = ( rng.random( ( max( 4 * nb_points, 1024 ), 3 ) ) - 0.5 ) * e
            uv = self.project_points( batch )
            iu = np.floor( ( uv[ ..., 0 ] - self.u_min ) / self.du ).astype( int )
            iv = np.floor( ( uv[ ..., 1 ] - self.v_min ) / self.dv ).astype( int )
            inside = ( iu >= 0 ) & ( iu < self.nb_u_host ) & ( iv >= 0 ) & ( iv < self.nb_v_host )
            v = np.where( inside, vals[ np.arange( nb_angles )[ :, None ], iu.clip( 0, self.nb_u_host - 1 ),
                                        iv.clip( 0, self.nb_v_host - 1 ) ], 0.0 )
            ok = np.all( v > threshold, axis = 0 )
            pts.append( batch[ ok ] )
            tried += len( batch )
            if tried > 200 * nb_points + 1e6:
                raise ValueError( "visual_hull_points : l'enveloppe visuelle est ( presque ) vide dans ce cube" )
        return np.concatenate( pts )[ :nb_points ]

    # -- accumulation ------------------------------------------------------

    def add_sphere( self, center, radius: float, density: float = 1.0 ) -> "Radiographs":
        """Ajoute la projection d'une boule uniforme.

        L'intégrale le long d'un rayon passant à distance ρ de l'axe de la boule est la corde
        `ρ_m · 2·√(r² − ρ²)` ( nulle hors du disque projeté ), où ρ² = (u − u0)² + (v − v0)² et
        ( u0, v0 ) le centre projeté. Elle est intégrée sur chaque pixel par une quadrature de
        Gauss-Legendre ( `quadrature` points par axe -- exacte partout sauf sur les pixels que le
        bord du disque traverse, où la corde n'est pas polynomiale ), et la valeur stockée est la
        densité moyenne sur le pixel ( `mass = value · du · dv` ). La masse totale par angle vaut
        donc `4/3 π r³ ρ_m` aux erreurs de quadrature près, tant que la boule tient dans le
        détecteur. Ce n'est pas l'intégrale exacte de `Sinogram.add_disk` : en 2D l'intégrale de la
        corde sur un rectangle n'a pas de forme close commode, et une reconstruction par transport
        normalise de toute façon chaque angle à la masse 1.

        Retourne self pour permettre le chaînage.
        """
        center = np.asarray( center, dtype = float )
        if center.shape != ( 3, ):
            raise ValueError( "center doit être de shape [ 3 ]" )
        if radius <= 0:
            raise ValueError( "radius doit être > 0" )

        r = float( radius )
        uv0 = self.project_points( center[ None, : ] )[ :, 0, : ]                  # [ nb_angles, 2 ]

        # les noeuds de Gauss-Legendre sur [ 0, 1 ], et leurs poids ( somme 1 )
        x, w = np.polynomial.legendre.leggauss( self.quadrature )
        x, w = ( x + 1 ) / 2, w / 2
        us = self.u_edges[ :-1, None ] + self.du * x[ None, : ]                   # [ nb_u, q ]
        vs = self.v_edges[ :-1, None ] + self.dv * x[ None, : ]                   # [ nb_v, q ]

        contribution = np.zeros( ( len( self.angles ), self.nb_u_host, self.nb_v_host ) )
        for k in range( len( self.angles ) ):
            a2 = ( us - uv0[ k, 0 ] ) ** 2                                         # [ nb_u, q ]
            b2 = ( vs - uv0[ k, 1 ] ) ** 2                                         # [ nb_v, q ]
            chord = 2 * np.sqrt( np.maximum( r * r - a2[ :, :, None, None ] - b2[ None, None, :, : ], 0.0 ) )
            contribution[ k ] = np.einsum( "iajb,a,b->ij", chord, w, w )

        self.values = self.values + density * contribution
        return self

    # -- consommation ------------------------------------------------------

    def _image_kwargs( self ):
        return dict( origin = [ self.u_min, self.v_min ],
                     frame = [ [ self.du, 0.0 ], [ 0.0, self.dv ] ] )

    def image( self, k: int, background: float = 0.0 ) -> Image:
        """`Image` 2D de la radiographie à l'angle k, en coordonnées détecteur réelles.

        `background` : une densité ajoutée PARTOUT ( en fraction de la valeur moyenne de l'angle )
        -- ce qu'un transport semi-discret demande pour qu'aucune cellule de Laguerre ne soit de
        mesure nulle ( un dirac projeté hors de l'ombre de l'objet n'aurait sinon aucun gradient,
        voir `OtPlan` ). Une radiographie de boules est nulle hors de leurs ombres.
        """
        vals = np.asarray( self.values )[ k ]
        if background:
            vals = vals + background * float( vals.mean() )
        return Image( values = vals, **self._image_kwargs() )

    def batched_image( self, background: float = 0.0 ) -> Image:
        """Toutes les radiographies d'un coup : une `Image` batchée sur `num_angle`."""
        vals = np.asarray( self.values )
        if background:
            vals = vals + background * vals.mean( axis = ( 1, 2 ), keepdims = True )
        return Image( values = vals, batch_axes = [ self.num_angle ], **self._image_kwargs() )

    def mass( self, k: int | None = None ):
        """Masse totale ( ∫∫ radiographie du dv ) à l'angle k, ou par angle si k est None."""
        m = np.asarray( self.values ).sum( axis = ( 1, 2 ) ) * self.du * self.dv
        return m if k is None else m[ k ]
