"""Le HALO : la matière HORS du champ de vue, et son empreinte sur le sinogramme.

Quand la pièce observée est plus large que le détecteur, chaque profil mesuré contient la
contribution de matière qui n'est PAS reconstructible : à l'angle θ, un point de rayon r > S
(S = extent/2) n'est vu que sur la fenêtre angulaire `2·arcsin( S/r )`. La masse mesurée
`∫p_θ` varie donc avec l'angle, alors que `OtPlan1d` normalise les deux distributions à masse 1 :
l'excédent est redistribué DANS le champ, et vient boucher les vides que les diracs/disques
étaient précisément là pour préserver.

Ce module retire cet excédent. Le parti pris, qui explique toute la suite :

    on ne cherche PAS à reconstruire l'extérieur, seulement son EMPREINTE `c( θ, s )`.

Les données couvrent toutes les droites rencontrant le disque de rayon S -- c'est le *problème
intérieur* de la transformée de Radon, dont le noyau n'est pas trivial : l'extérieur n'est pas
identifiable en détail. Le mailler finement fabriquerait des degrés de liberté qui iraient
absorber du vrai signal intérieur. Deux conséquences de conception :

- le halo est GROSSIER, et de plus en plus grossier vers l'extérieur (voir `_build_cells`) ;
- il n'est pas non plus une fonction libre de `( θ, s )` -- ce serait trop de DDL, et dégénéré
  avec l'intérieur. On le contraint à être la transformée de Radon d'une densité ≥ 0 supportée
  hors du disque de rayon S, ce qui impose GRATUITEMENT la cohérence angulaire (conditions de
  moments de Helgason-Ludwig). C'est cette contrainte qui rend l'estimation bien posée avec très
  peu de paramètres.

Ce qui rend le partage intérieur/extérieur identifiable en PRATIQUE alors qu'il ne l'est pas au
sens de l'opérateur : les deux a priori sont orthogonaux en échelle -- l'intérieur est parcimonieux
et veut laisser des vides, le halo est lisse et grossier.

`Halo` est FIXE (le maillage ne bouge pas, contrairement aux diracs) : la carte
`densités des cellules -> contribution au sinogramme` est donc un opérateur LINÉAIRE précalculé
une fois, et l'estimation est un moindres carrés POSITIF -- convexe, sans aucune non-convexité
ajoutée au problème.

`alternate` enchaîne les deux blocs (l'estimation du halo a besoin de l'intérieur, et
réciproquement) ; deux ou trois passes suffisent, justement parce qu'ils vivent à des échelles
différentes.

`Sinogram.debias_and_equalize_mass` est le cas dégénéré de ce module : un halo à UN degré de
liberté par angle (`c_k` constant en `s`), sans aucun lien géométrique entre les angles. Correct
quand l'objet est BEAUCOUP plus grand que la fenêtre, faux dès qu'il ne dépasse que d'un facteur
1.5-2 -- le cas gênant.
"""
import numpy as np
from scipy.optimize import nnls

from .disks import DiskProjector
from .Reconstruction import Reconstruction
from .Sinogram import Sinogram


class Halo:
    """Maillage log-polaire FIXE de l'extérieur du champ de vue, son opérateur de projection, et
    les densités ajustées.

    Géométrie (tout est en coordonnées MONDE, celles de `Sinogram`) :
    - `inner_radius` (défaut `extent/2`) : le bord du champ de vue -- rien du halo n'entre dedans,
      c'est le domaine réservé aux diracs/disques ;
    - `outer_radius` : jusqu'où va la matière. À surestimer plutôt qu'à sous-estimer : une cellule
      inutile prend le poids 0 (la positivité s'en charge), une cellule manquante ne peut pas être
      inventée ;
    - `growth` : rapport des rayons de deux anneaux consécutifs (épaisseur géométriquement
      croissante) ;
    - `nb_sectors` : nombre de secteurs que vaudrait un anneau situé EXACTEMENT en `inner_radius` ;
      les vrais anneaux en ont moins, de plus en plus en s'éloignant (voir `_build_cells`). Le
      défaut de 32 est le coude mesuré sur un objet extérieur compact (le cas le plus exigeant) :
      en dessous, le maillage ne localise plus l'objet et la masse visible par angle décroche ;
      au-dessus, l'erreur ne descend plus -- elle est alors limitée par la grille grossière.

    `nb_coarse_bins` : le halo est estimé sur une grille détecteur GROSSIÈRE (les cases mesurées
    y sont regroupées par paquets entiers). Ce n'est pas qu'une économie : c'est une régularisation
    de plus, et c'est cohérent avec la prémisse -- l'empreinte du halo n'a pas de structure fine.

    Les poids (`weights`, une densité par cellule) démarrent à 0 -- un halo neuf ne corrige rien.
    `fit` les met à jour ; toutes les méthodes de sortie (`values`, `corrected`, `mass`) les lisent.
    """

    def __init__( self, sinogram: Sinogram, outer_radius: float, *, inner_radius: float | None = None,
                  growth: float = 1.6, nb_sectors: int = 32, nb_coarse_bins: int = 256,
                  phi_per_bin: float = 4.0, nb_phi_max: int = 1 << 15 ) -> None:
        self.sinogram = sinogram
        self.inner_radius = float( inner_radius if inner_radius is not None else sinogram.extent / 2 )
        self.outer_radius = float( outer_radius )
        if self.outer_radius <= self.inner_radius:
            raise ValueError( f"outer_radius ({ self.outer_radius }) doit dépasser inner_radius ({ self.inner_radius })" )
        if growth <= 1.0:
            raise ValueError( "growth doit être > 1" )
        self.growth = float( growth )
        self.nb_sectors = max( 1, int( nb_sectors ) )
        self.phi_per_bin = float( phi_per_bin )
        self.nb_phi_max = max( 8, int( nb_phi_max ) )

        # géométrie détecteur, côté HÔTE (tout ce module est du numpy : ni différentiation ni jit --
        # le halo est une CONSTANTE du point de vue de l'optimisation intérieure)
        self.angles = np.asarray( sinogram.angles, dtype = float )
        self.nb_angles = int( self.angles.size )
        self.nb_bins = int( sinogram.nb_bins_host )
        self.s_min = float( sinogram.s_min )
        self.dw = float( sinogram.dw )

        # regroupement des cases mesurées en cases grossières -- un diviseur exact de `nb_bins`,
        # pour que le regroupement (moyenne) et son inverse (`np.repeat`) conservent la masse.
        self.group = self._group_size( nb_coarse_bins )
        self.nb_coarse = self.nb_bins // self.group
        self.coarse_dw = self.dw * self.group
        self.coarse_edges = self.s_min + self.coarse_dw * np.arange( self.nb_coarse + 1 )

        self.cells = self._build_cells()                                   # [ ( a, b, phi0, phi1, ring ) ]
        #: `[ nb_cells, nb_angles, nb_coarse ]` -- densité projetée d'une cellule de densité 1
        self.operator = np.stack( [ self._cell_values( *c[ :4 ] ) for c in self.cells ] )
        #: aire de chaque cellule, `[ nb_cells ]` (masse d'une cellule = aire x densité)
        self.areas = np.array( [ 0.5 * ( b * b - a * a ) * ( p1 - p0 ) for a, b, p0, p1, _ in self.cells ] )
        #: densité ajustée par cellule, `[ nb_cells ]` -- 0 tant que `fit` n'a pas tourné
        self.weights = np.zeros( len( self.cells ) )

    def __repr__( self ) -> str:
        return ( f"Halo( { self.nb_cells } cellules, r { self.inner_radius:.3g}..{ self.outer_radius:.3g}, "
                 f"{ self.nb_coarse } cases grossières )" )

    @property
    def nb_cells( self ) -> int:
        return len( self.cells )

    # -- géométrie ---------------------------------------------------------

    def _group_size( self, nb_coarse_bins: int ) -> int:
        """Le plus petit diviseur de `nb_bins` donnant au plus `nb_coarse_bins` cases grossières."""
        target = max( 1, int( np.ceil( self.nb_bins / max( 1, int( nb_coarse_bins ) ) ) ) )
        for g in range( target, self.nb_bins + 1 ):
            if self.nb_bins % g == 0:
                return g
        return self.nb_bins

    def _build_cells( self ):
        """Anneaux d'épaisseur géométrique, découpés en secteurs de moins en moins nombreux.

        La loi de décroissance vient de la géométrie d'acquisition : un point de rayon r n'est vu
        que sur une fenêtre angulaire `2·arcsin( S/r ) ≈ 2S/r`, donc le nombre de mesures touchant
        l'anneau de rayon r décroît en `1/r` pendant que sa circonférence croît en `r` -- la taille
        de cellule angulaire doit croître au moins comme `r²`, d'où `nb_secteurs ∝ ( S/r )²`.

        Conséquence chiffrée : même sur un domaine dix fois plus large que le champ de vue, le halo
        reste à quelques dizaines de DDL. C'est voulu (voir la docstring du module).
        """
        cells, ring, a = [], 0, self.inner_radius
        while a < self.outer_radius * ( 1 - 1e-12 ):
            b = min( a * self.growth, self.outer_radius )
            r_mid = 0.5 * ( a + b )
            n = max( 1, int( round( self.nb_sectors * ( self.inner_radius / r_mid ) ** 2 ) ) )
            for k in range( n ):
                cells.append( ( a, b, 2 * np.pi * k / n, 2 * np.pi * ( k + 1 ) / n, ring ) )
            a, ring = b, ring + 1
        return cells

    # -- opérateur de projection -------------------------------------------

    def _cell_values( self, a: float, b: float, phi0: float, phi1: float ) -> np.ndarray:
        """Densité projetée `[ nb_angles, nb_coarse ]` d'une cellule (secteur d'anneau) de densité 1.

        EXACT dans la direction radiale, quadrature dans la direction angulaire. À φ fixé, le
        segment radial `r ∈ [a,b]` se projette sur `s = r·c` avec `c = cos( θ − φ )` ; l'élément de
        masse `r dr dφ` devient `( s/c )( ds/c ) dφ`, soit la densité EXACTE `|s|/c²·dφ` sur
        l'intervalle borné par `a·c` et `b·c` (masse totale `dφ( b² − a² )/2`, l'aire de la
        tranche -- c'est la vérification du calcul). Intégrer analytiquement en r évite le facteur
        `n_r` de coût d'une quadrature 2D, et rend exacte la singularité en `√( a² − s² )` du bord
        interne, qui tombe pile aux extrémités du détecteur.

        Il reste à sommer ces rampes `α|s|` sur les cases. Les déposer case par case coûterait le
        nombre de cases couvertes ; on passe donc par la PRIMITIVE, dont chaque rampe ne modifie
        que deux indices d'arête. Avec `β = sign(c)·dφ/( 2c² )`, `lo = min( a·c, b·c )` et
        `hi = max( a·c, b·c )`, la primitive d'UNE rampe s'écrit sans indicatrice :

            F( x ) = β·( clip( x, lo, hi )² − lo² )

        (nulle en `x ≤ lo`, constante `β( hi² − lo² )` = la masse en `x ≥ hi`). En séparant les
        rampes finies / actives / pas encore commencées, la somme sur toutes les rampes vaut
        `F( x ) = C( x ) + Q( x )·x² − L( x )` avec `Q = Σ_{lo≤x≤hi} β`, `L = Σ_{lo≤x} β·lo²` et
        `C = Σ_{hi<x} β·hi²` -- trois fonctions en ESCALIER, donc trois `bincount` sur les indices
        d'arête suivis d'un `cumsum`. Coût O(1) par nœud, indépendant du nombre de cases.

        Le saut de chacune est porté par la PREMIÈRE ARÊTE À DROITE de sa position (`ceil`, pas
        `round` : arrondir déplacerait une rampe sur deux d'une demi-case, et l'erreur, propagée
        par le `cumsum` puis multipliée par `x²`, contaminerait tout le profil). C'est ce choix qui
        rend le résultat EXACT et non approché : `L` et `C` gardant les vrais `lo²`/`hi²` et non
        des valeurs arrondies, on a `F( e ) = β( e² − lo² )` à l'arête près de `lo`, donc la masse
        de chaque case est l'intégrale exacte de la rampe sur cette case.

        `β` explose quand `c → 0` (le segment se projette en un point). Rien ne DIVERGE pour
        autant -- `β·lo² = dφ·a²/2`, `β·hi² = dφ·b²/2` et `β·x²` restent finis, `x` étant coincé
        entre `lo` et `hi` -- mais `F = C + Q·x² − L` devient une différence de termes énormes,
        et l'annulation catastrophique qui s'ensuit ruine tout le profil (mesuré : masse totale
        négative pour certaines valeurs de `phi_per_bin`, celles qui alignent un nœud sur
        `θ ± π/2`). On borne donc `|c|` par en dessous, à la valeur qui rend la rampe mille fois
        plus étroite qu'une case : au-delà sa position exacte n'a plus aucun sens à cette
        résolution, et `β` reste dans une plage où la soustraction est exacte.

        Les indices sont écrêtés dans `[ 0, nb_coarse+1 ]` : une rampe entièrement à GAUCHE du
        détecteur s'achève à l'arête 0 (masse déjà écoulée, aucune case touchée), une rampe
        entièrement à DROITE atterrit dans l'index de débordement `nb_coarse+1`, jamais lu.
        """
        # pas angulaire : `phi_per_bin` extrémités de rampe par case grossière (le nœud le plus
        # externe de la cellule, en `b`, fixe le pas). En dessous, la somme des rampes ondule à
        # l'échelle de la case, alors que l'empreinte du halo, elle, est lisse.
        n_phi = int( np.clip( np.ceil( self.phi_per_bin * b * ( phi1 - phi0 ) / self.coarse_dw ),
                              8, self.nb_phi_max ) )
        dphi = ( phi1 - phi0 ) / n_phi
        phi = phi0 + dphi * ( np.arange( n_phi ) + 0.5 )                   # [ n_phi ]

        c = np.cos( self.angles[ :, None ] - phi[ None, : ] )              # [ nb_angles, n_phi ]
        c_min = 1e-3 * self.coarse_dw / max( b - a, 1e-12 )                # cf. docstring
        c = np.where( np.abs( c ) < c_min, np.where( c < 0, -c_min, c_min ), c )
        beta = np.sign( c ) * dphi / ( 2 * c * c )
        lo = np.minimum( a * c, b * c )
        hi = np.maximum( a * c, b * c )

        W = self.nb_coarse + 2                                             # +1 arête, +1 débordement
        base = np.arange( self.nb_angles )[ :, None ] * W
        ml = self.nb_angles * W

        def edge_of( pos ):
            """Index (aplati) de la première arête à droite de `pos`."""
            x = np.ceil( ( pos - self.s_min ) / self.coarse_dw )
            return ( base + np.clip( x, 0, W - 1 ).astype( int ) ).ravel()

        i_lo, i_hi, bt = edge_of( lo ), edge_of( hi ), beta.ravel()
        Q = np.bincount( i_lo, bt, ml ) - np.bincount( i_hi, bt, ml )
        L = np.bincount( i_lo, ( beta * lo * lo ).ravel(), ml )
        C = np.bincount( i_hi, ( beta * hi * hi ).ravel(), ml )

        n = self.nb_coarse + 1
        Q = np.cumsum( Q.reshape( self.nb_angles, W ), axis = 1 )[ :, :n ]
        L = np.cumsum( L.reshape( self.nb_angles, W ), axis = 1 )[ :, :n ]
        C = np.cumsum( C.reshape( self.nb_angles, W ), axis = 1 )[ :, :n ]

        F = C + Q * self.coarse_edges ** 2 - L                             # primitive aux arêtes
        return ( F[ :, 1: ] - F[ :, :-1 ] ) / self.coarse_dw               # masse par case -> densité

    # -- estimation --------------------------------------------------------

    def _smoothness_rows( self ) -> np.ndarray:
        """Différences entre secteurs ANGULAIREMENT voisins d'un même anneau (cycliques).

        Pas de couplage radial : les anneaux n'ont pas le même nombre de secteurs, l'appariement
        serait arbitraire -- et la positivité + la grossièreté du maillage régularisent déjà
        beaucoup dans cette direction.
        """
        rings: dict[ int, list[ int ] ] = {}
        for i, ( _, _, _, _, ring ) in enumerate( self.cells ):
            rings.setdefault( ring, [] ).append( i )
        rows = []
        for idx in rings.values():
            if len( idx ) < 2:
                continue
            for k, i in enumerate( idx ):
                row = np.zeros( self.nb_cells )
                row[ i ] = 1.0
                row[ idx[ ( k + 1 ) % len( idx ) ] ] = -1.0
                rows.append( row )
        return np.array( rows ) if rows else np.zeros( ( 0, self.nb_cells ) )

    def fit( self, residual, *, target_mass = None, mass_weight: float = 10.0,
             ridge: float = 1e-3, smooth: float = 3e-2 ) -> "Halo":
        """Ajuste les densités des cellules sur le `residual` (mesuré MOINS modèle intérieur),
        `[ nb_angles, nb_bins ]` en DENSITÉ. Met `weights` à jour et renvoie `self`.

        Le résidu est regroupé sur la grille grossière avant l'ajustement -- c'est là qu'on gagne
        le plus : le bruit du résidu (sous-échantillonnage du nuage, cf. `interior_values`) est
        blanc entre cases, et une régression à quelques dizaines de DDL sur `nb_angles x nb_coarse`
        équations le divise d'un facteur `√( nb_angles·nb_coarse / nb_cells )`, typiquement 70.

        `target_mass` (`[ nb_angles ]`, optionnel) : la masse que le halo doit rendre VISIBLE à
        chaque angle, soit `∫p_θ − M_in`. C'est l'ancrage le plus solide du problème -- il vient
        directement de la donnée, sans passer par la forme des profils -- et il conditionne
        nettement mieux l'ajustement. `mass_weight` en règle le poids relatif.

        `ridge` / `smooth` : Tikhonov, et lissage angulaire intra-anneau (`_smoothness_rows`).
        Chaque bloc est normalisé par sa propre norme de Frobenius, donc les trois poids sont
        sans dimension et comparables entre eux.

        La résolution est un moindres carrés POSITIF (`scipy.optimize.nnls`) : la positivité n'est
        pas cosmétique, c'est elle qui empêche le halo d'aller creuser du signal intérieur.
        """
        res = np.asarray( residual, dtype = float )
        if res.shape != ( self.nb_angles, self.nb_bins ):
            raise ValueError( f"residual doit être de shape [ { self.nb_angles }, { self.nb_bins } ], "
                              f"reçu { res.shape }" )
        coarse = res.reshape( self.nb_angles, self.nb_coarse, self.group ).mean( axis = 2 )

        blocks = [ ( self.operator.reshape( self.nb_cells, -1 ).T, coarse.ravel(), 1.0 ) ]

        if target_mass is not None:
            # masse visible par angle d'une cellule de densité 1 : [ nb_angles, nb_cells ]
            vis = self.operator.sum( axis = 2 ).T * self.coarse_dw
            blocks.append( ( vis, np.asarray( target_mass, dtype = float ).ravel(), float( mass_weight ) ) )
        if ridge > 0:
            blocks.append( ( np.eye( self.nb_cells ), np.zeros( self.nb_cells ), float( ridge ) ) )
        if smooth > 0:
            rows = self._smoothness_rows()
            if rows.size:
                blocks.append( ( rows, np.zeros( len( rows ) ), float( smooth ) ) )

        mats, rhs = [], []
        for mat, vec, lam in blocks:
            norm = np.linalg.norm( mat )
            if norm == 0 or lam == 0:
                continue
            mats.append( mat * ( lam / norm ) )
            rhs.append( vec * ( lam / norm ) )

        self.weights = nnls( np.concatenate( mats ), np.concatenate( rhs ) )[ 0 ]
        return self

    # -- sorties -----------------------------------------------------------

    def values( self ) -> np.ndarray:
        """Empreinte du halo sur le sinogramme, `[ nb_angles, nb_bins ]`, en densité.

        Le retour à la grille fine est une simple répétition (`np.repeat`) : constante par paquet,
        donc de masse exacte. L'escalier qu'elle introduit est du deuxième ordre -- l'empreinte
        varie peu d'une case grossière à la suivante, c'est toute la prémisse du module.
        """
        coarse = np.tensordot( self.weights, self.operator, axes = ( 0, 0 ) )
        return np.repeat( coarse, self.group, axis = 1 )

    def visible_mass( self ) -> np.ndarray:
        """Masse du halo tombant DANS le détecteur, par angle, `[ nb_angles ]`."""
        return self.values().sum( axis = 1 ) * self.dw

    def mass( self ) -> float:
        """Masse totale du halo (dont une partie n'est visible à aucun angle)."""
        return float( self.weights @ self.areas )

    def corrected( self, sinogram: Sinogram | None = None ) -> Sinogram:
        """Le sinogramme débarrassé de l'empreinte du halo, écrêté à 0.

        L'écrêtage rompt légèrement la comptabilité de masse ; c'est sans conséquence ici (les
        valeurs concernées sont du bruit autour de zéro) et ça garantit une densité cible valide
        pour `OtPlan1d`.
        """
        sino = sinogram if sinogram is not None else self.sinogram
        out = Sinogram( nb_angles = self.nb_angles, nb_bins = self.nb_bins,
                        extent = sino.extent, detector_center = sino.detector_center )
        out.values = np.clip( np.asarray( sino.values ) - self.values(), 0.0, None )
        return out


# -- projection du nuage intérieur -----------------------------------------


def interior_values( sinogram: Sinogram, points, mass: float, radius: float | None = None,
                     max_points: int | None = None, seed: int = 0 ) -> np.ndarray:
    """Densité `[ nb_angles, nb_bins ]` projetée par le nuage `points`, portant la masse totale
    `mass` à chaque angle.

    `mass` est fournie de l'EXTÉRIEUR parce que le nuage n'en a pas : `OtPlan1d` normalise ses deux
    distributions, donc la reconstruction ne fixe que la FORME. C'est `alternate` qui décide de la
    masse intérieure (voir sa docstring).

    `radius` : `None` pour des diracs (déposés linéairement sur les deux cases voisines, masse
    conservée), sinon des disques de ce rayon (`DiskProjector`, la même projection que celle que
    minimise `DiskModel`).

    `max_points` : le nuage est SOUS-ÉCHANTILLONNÉ au-delà. Ce résidu ne sert qu'à donner une forme
    au halo, à sa résolution grossière et via une régression à quelques dizaines de DDL : y mettre
    les 1e7 diracs d'une reconstruction fine coûterait cher pour rien (cf. `Halo.fit`).
    """
    pts = np.asarray( points, dtype = float )
    if pts.ndim != 2 or pts.shape[ 1 ] != 2:
        raise ValueError( f"points doit être de shape [ n, 2 ], reçu { pts.shape }" )
    n = len( pts )
    if n == 0:
        return np.zeros( ( int( sinogram.nb_angles.value ), sinogram.nb_bins_host ) )
    if max_points is not None and n > max_points:
        pts = pts[ np.random.default_rng( seed ).choice( n, max_points, replace = False ) ]

    if radius is not None:
        # densité d'un disque de densité 1 -> on renormalise pour que chaque disque porte mass/len
        vals = np.asarray( DiskProjector( sinogram, radius = radius ).values( pts ) )
        return vals * ( mass / ( len( pts ) * np.pi * radius * radius ) )

    nb_angles, nb_bins = len( sinogram.angles ), sinogram.nb_bins_host
    out = np.zeros( nb_angles * nb_bins )
    per_point = mass / ( len( pts ) * sinogram.dw )                        # densité, pas masse

    # tranches d'angles : le tableau des positions projetées fait [ nb_angles, nb_points ], vite
    # plus gros que tout le reste (600 angles x 5e5 points = 2.4 Go).
    chunk = max( 1, int( 2e7 // max( 1, len( pts ) ) ) )
    for k0 in range( 0, nb_angles, chunk ):
        normals = sinogram.normals[ k0 : k0 + chunk ]                      # [ na, 2 ]
        s = normals @ pts.T                                                # [ na, nb_points ]
        x = ( s - sinogram.s_min ) / sinogram.dw - 0.5                     # en indices de CENTRES
        i0 = np.floor( x ).astype( int )
        w1 = x - i0
        base = ( k0 + np.arange( len( normals ) ) )[ :, None ] * nb_bins
        for idx, w in ( ( i0, 1.0 - w1 ), ( i0 + 1, w1 ) ):
            ok = ( idx >= 0 ) & ( idx < nb_bins )
            flat = ( base + np.clip( idx, 0, nb_bins - 1 ) )[ ok ]
            out += np.bincount( flat, ( w * per_point )[ ok ], nb_angles * nb_bins )
    return out.reshape( nb_angles, nb_bins )


# -- l'alternance ----------------------------------------------------------


def alternate( sinogram: Sinogram, solve, *, outer_radius: float | None = None,
               halo: Halo | None = None, nb_outer: int = 3, nb_points: int | None = None,
               interior_mass: float | None = None, radius: float | None = None,
               max_residual_points: int | None = 500_000, verbose: bool = False,
               halo_kwargs: dict | None = None,
               **recon_kwargs ) -> tuple[ Reconstruction, Halo ]:
    """Alterne estimation du HALO et reconstruction de l'INTÉRIEUR, et renvoie les deux.

    « retirer du sinogramme ce qui a été trouvé à l'extérieur » est circulaire pris au pied de la
    lettre : pour connaître l'extérieur il faut connaître l'intérieur. On alterne donc, en partant
    d'un halo NUL -- la première reconstruction est la mauvaise (les vides sont bouchés), mais
    l'intérieur, confiné au champ de vue et de capacité limitée, ne peut pas reproduire un
    sinogramme angulairement incohérent : ce qu'il laisse au résidu EST la part inexplicable de la
    donnée, exactement le signal que cherche le halo. Deux ou trois passes suffisent, les deux
    modèles vivant à des échelles différentes.

        rec, halo = alternate( sino, lambda r: r.multiscale( 5000 ), outer_radius = 4.0 )

    `solve( rec )` : UNE résolution intérieure complète, partant du nuage courant de `rec` (qui est
    donc réchauffé d'une passe à l'autre) et sur le sinogramme corrigé du moment. Doit renvoyer
    `rec` -- typiquement `lambda r: r.multiscale( n )` ou `lambda r: r.diracs().disks( radius )`.

    `interior_mass` : la masse à attribuer à l'INTÉRIEUR. Par défaut `min_θ ∫p_θ`, qui est la
    borne exacte (`∫p_θ = M_in + M_out( θ )` avec `M_out ≥ 0`) -- et le seul choix qui garantisse
    un résidu de masse positive à tous les angles, donc quelque chose à ajuster pour un halo
    contraint positif. C'est une borne SUPÉRIEURE, atteinte seulement s'il existe un angle où
    l'objet tient entièrement dans le détecteur ; sinon le halo est sous-estimé, et c'est le
    paramètre à baisser. `mass_profile` donne la courbe `∫p_θ` pour en juger, et `void_fraction`
    le critère : le bon partage est celui qui maximise le vide.

    `radius` : rayon des disques pour la projection du nuage dans le résidu (`None` = diracs). À
    accorder au modèle que joue `solve`.

    `nb_points` : tirage initial, pour un `solve` qui ne s'en charge pas lui-même (`multiscale` le
    fait, `diracs`/`disks` non).

    `recon_kwargs` -> `Reconstruction`. `extent` y vaut par défaut celle du DÉTECTEUR et non celle
    de l'objet : l'intérieur doit rester dans le champ de vue, c'est le halo qui porte le reste.
    """
    if halo is None:
        if outer_radius is None:
            raise ValueError( "fournir `halo`, ou `outer_radius` pour en construire un" )
        halo = Halo( sinogram, outer_radius = outer_radius, **( halo_kwargs or {} ) )

    raw = np.asarray( sinogram.values, dtype = float )
    per_angle = raw.sum( axis = 1 ) * sinogram.dw
    m_in = float( per_angle.min() ) if interior_mass is None else float( interior_mass )
    target = np.maximum( per_angle - m_in, 0.0 )                           # masse VISIBLE du halo

    if verbose:
        print( f"[halo] { halo }" )
        print( f"[halo] masse par angle : min={ per_angle.min():.4g} max={ per_angle.max():.4g} "
               f"-> M_in={ m_in:.4g}, halo visible <= { target.max():.4g}" )

    rec = Reconstruction( sinogram, **recon_kwargs )
    if nb_points is not None:
        rec.random_points( nb_points )
    for it in range( max( 1, int( nb_outer ) ) ):
        rec.set_sinogram( sinogram if it == 0 else halo.corrected( sinogram ) )
        solve( rec )
        if it + 1 >= nb_outer:
            break

        fwd = interior_values( sinogram, rec.positions, m_in, radius = radius,
                               max_points = max_residual_points, seed = it )
        halo.fit( raw - fwd, target_mass = target )
        if verbose:
            got = halo.visible_mass()
            print( f"[halo] passe { it }: masse halo { halo.mass():.4g} "
                   f"(visible { got.min():.4g}..{ got.max():.4g}, cible { target.min():.4g}..{ target.max():.4g}), "
                   f"{ int( ( halo.weights > 0 ).sum() ) }/{ halo.nb_cells } cellules actives" )

    return rec, halo


def scan_interior_mass( halo: Halo, points, masses = None, radius: float | None = None,
                        max_points: int | None = 500_000, sinogram: Sinogram | None = None,
                        **fit_kwargs ) -> dict:
    """Balaie la masse intérieure `M_in` et renvoie, pour chacune, ce que le halo en fait.

    `M_in` est le paramètre le moins déterminé du problème (voir `alternate`) : la borne par défaut
    `min_θ ∫p_θ` n'est atteinte que s'il existe un angle où l'objet tient entièrement dans le
    détecteur, ce qui est faux dès que la pièce déborde dans TOUTES les directions. Mesuré sur
    `experiments/halo_demo`, elle surestime alors `M_in` de 50%, et le halo récupère 1.5 de masse
    au lieu de 6.8 -- alors qu'à la bonne valeur il la retrouve à 0.8% près. C'est de loin la
    première source d'erreur du module, avant la finesse du maillage.

    Le balayage est BON MARCHÉ : la projection de l'intérieur est linéaire en `M_in`, on ne la
    calcule donc qu'une fois, et chaque point du balayage n'est qu'un NNLS à quelques dizaines
    d'inconnues. Renvoie un dict de tableaux (`masses`, `halo_mass`, `dispersion`, `nb_active`)
    -- `dispersion` étant l'écart-type relatif de `∫q_θ` après correction, dont le minimum encadre
    la bonne valeur (creux peu marqué : à utiliser comme indice, pas comme estimateur).

    À croiser avec `void_fraction` sur la reconstruction obtenue, qui est le vrai critère : le bon
    partage est celui qui rend les vides les plus vides.
    """
    sino = sinogram if sinogram is not None else halo.sinogram
    raw = np.asarray( sino.values, dtype = float )
    per_angle = raw.sum( axis = 1 ) * sino.dw
    if masses is None:
        masses = np.linspace( 0.5, 1.0, 11 ) * per_angle.min()
    masses = np.asarray( masses, dtype = float )

    unit = interior_values( sino, points, 1.0, radius = radius, max_points = max_points )
    keep, out = halo.weights, { k: [] for k in ( "halo_mass", "dispersion", "nb_active" ) }
    for m in masses:
        halo.fit( raw - m * unit, target_mass = np.maximum( per_angle - m, 0.0 ), **fit_kwargs )
        q = np.clip( raw - halo.values(), 0.0, None ).sum( axis = 1 ) * sino.dw
        out[ "halo_mass" ].append( halo.mass() )
        out[ "dispersion" ].append( float( q.std() / max( q.mean(), 1e-30 ) ) )
        out[ "nb_active" ].append( int( ( halo.weights > 0 ).sum() ) )
    halo.weights = keep                                                    # le balayage n'ajuste rien
    return dict( masses = masses, **{ k: np.array( v ) for k, v in out.items() } )


# -- diagnostics -----------------------------------------------------------


def mass_profile( sinogram: Sinogram ) -> np.ndarray:
    """`∫p_θ` par angle, `[ nb_angles ]` -- la mesure DIRECTE de la fuite.

    Constante à la précision du bruit = l'objet tient dans le détecteur, rien à corriger. Sa
    variation est exactement `M_out( θ )`, la masse extérieure vue à l'angle θ.
    """
    return np.asarray( sinogram.mass(), dtype = float )


def void_fraction( points, extent: float, nb_cells: int | None = None, center = ( 0.0, 0.0 ) ) -> float:
    """Fraction des cases d'une grille `nb_cells²` (couvrant `extent` autour de `center`) que le
    nuage laisse VIDES -- le critère qui motive tout ce module.

    Diagnostic COMPARATIF : à nuage de même taille et même grille, plus il est haut, mieux les
    vides ont été préservés. Il compte aussi le fond hors objet, et n'a de sens qu'à `nb_cells`
    ÉGAL -- d'où le défaut `√n` : trop fin, la grille sature (`n` points ne peuvent occuper que
    `n` cases sur `nb_cells²`, tout nuage y paraît également vide) ; trop grossier, tout est
    occupé. À `√n` cases de côté, un nuage bien étalé en remplit environ 63%.
    """
    pts = np.asarray( points, dtype = float )
    nb_cells = max( 2, int( np.sqrt( len( pts ) ) ) ) if nb_cells is None else int( nb_cells )
    lo = np.asarray( center, dtype = float ) - extent / 2
    idx = np.floor( ( pts - lo ) / extent * nb_cells ).astype( int )
    ok = np.all( ( idx >= 0 ) & ( idx < nb_cells ), axis = 1 )
    flat = idx[ ok, 0 ] * nb_cells + idx[ ok, 1 ]
    return float( 1.0 - np.count_nonzero( np.bincount( flat, minlength = nb_cells ** 2 ) ) / nb_cells ** 2 )
