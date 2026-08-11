"""Reconstruction sur un MAILLAGE gradué couvrant tout l'objet, détecteur compris et débordé.

C'est l'étape qui précède les diracs quand la pièce est plus large que le détecteur. Le
raisonnement, en trois temps :

1. on résout d'abord TOUT sur un maillage -- l'intérieur du champ de vue en cellules fines,
   l'extérieur en cellules de plus en plus grosses. Problème CONVEXE, donc pas de minimum local,
   pas d'initialisation à soigner ;
2. le maillage donne alors le partage intérieur/extérieur de la masse, qui n'était PAS accessible
   autrement. C'est ce qui règle le point dur de `halo.py` : là-bas `M_in` devait être deviné
   (borné par `min_θ ∫p_θ`, borne fausse de 50% dès qu'aucun angle ne voit l'objet entier), ici il
   se lit sur la solution ;
3. on RETIRE du sinogramme la contribution des cellules EXTÉRIEURES, et on rend l'intérieur aux
   diracs/disques, dont c'est le métier -- eux seuls savent laisser des vides. Le maillage, lui,
   est trop lisse pour ça : c'est un estimateur de fond, pas de structure.

La graduation vient de la géométrie d'acquisition : un point de rayon r n'est vu que sur la
fenêtre angulaire `2·arcsin( S/r )`, donc l'information disponible s'effondre en s'éloignant. Des
cellules fines là-bas ne feraient qu'absorber du signal intérieur, et la régularisation prend le
relais.

Elle est QUADRATIQUE par défaut (voir `solve`), donc le problème reste LINÉAIRE et se résout par
gradient conjugué. C'est le bon compromis ici parce que le maillage n'a pas à résoudre les détails
intérieurs -- c'est le métier des diracs : il doit livrer l'empreinte extérieure et le partage de
masse, deux quantités lisses. La variation totale reste disponible, et donne une carte plus propre
au bord du champ de vue, mais rend le problème non linéaire pour huit fois le temps de calcul.

== Comment l'opérateur est calculé (et pourquoi il n'y a pas de matrice)

Toutes les cellules sont des CARRÉS, seule leur taille change. Or la projection d'un carré de côté
h est un trapèze analytique qui ne dépend que de `( h, θ )` -- pas de la position, qui ne fait que
le DÉCALER. La projection d'un niveau de maillage est donc :

    projeter = déposer les poids aux positions projetées des centres, puis CONVOLUER par le trapèze

Le dépôt est une matrice creuse fixe (deux cases par cellule et par angle, dépôt linéaire), la
convolution une FFT par angle avec un noyau précalculé. Aucune matrice système : la mémoire est en
`O( nb_cellules x nb_angles )` au lieu de `O( nb_cellules x nb_angles x nb_cases )`. Le trapèze
étant SYMÉTRIQUE, la convolution est auto-adjointe et l'adjoint de l'opérateur complet est
exactement le dépôt transposé -- pas d'adjoint approché, donc un solveur qui converge vraiment.
"""
import numpy as np
import scipy.sparse as sp
from scipy.fft import irfft, next_fast_len, rfft
from scipy.spatial import cKDTree

from .Sinogram import Sinogram


def scan_exterior_scale( mesh, solve, alphas = None, *, sinogram: Sinogram | None = None,
                         metrics = None, verbose: bool = False, **recon_kwargs ) -> dict:
    """Balaie `alpha`, le facteur appliqué à l'empreinte extérieure avant soustraction, et
    reconstruit en diracs pour chacun.

    Pourquoi un balayage plutôt qu'un calcul : le noyau du problème intérieur autorise à déplacer
    un niveau lentement variable entre le dedans et le dehors, et rien dans les données ne dit
    lequel choisir -- c'est la définition d'un noyau. Toute régularisation en choisit un, et on a
    mesuré (voir `solve`) que ce choix vaut environ 10% de la masse intérieure. Plutôt que de le
    cacher dans un poids de régularisation, on l'expose comme UN scalaire, qu'on peut confronter à
    ce dont on dispose par ailleurs.

    `solve( rec )` : une résolution intérieure complète, comme pour `halo.alternate`.
    `metrics` : `{ nom: f( positions ) -> float }` en plus de ceux calculés d'office
    (`void_fraction`, la masse restante par angle, et `clipped_fraction`, qui dit à partir de quel
    `alpha` la soustraction devient physiquement impossible).

    Renvoie un dict de tableaux plus `clouds`, la liste des nuages obtenus.

    MESURÉ, et c'est une mise en garde : sur `experiments/lung_mesh`, l'optimum vrai est net
    (erreur de densité minimale à α = 0.90, là où la masse intérieure retombe sur 1225 pour 1226)
    mais AUCUN indicateur observable ne le trouve. Le vide croît de façon monotone (33% à 43% entre
    α = 0.9 et 1.1), l'écrêtage reste nul jusqu'à α = 1, et le résidu OT ne fait que croître. C'est
    la définition même d'un noyau : rien dans les données ne distingue ces solutions. Ce balayage
    sert donc à EXPLOITER un ancrage extérieur (densité matériau connue, support plus serré,
    quelques vues grand champ), pas à s'en passer. Consolation : l'erreur est plate autour de
    l'optimum -- 0.450 à α = 0.90 contre 0.471 à α = 1 --, donc le défaut α = 1 reste raisonnable.
    Seul le résidu OT montre une amorce de coude vers l'optimum ; à confronter à d'autres cas
    avant d'en faire un critère.
    """
    from .halo import void_fraction                    # tardif : `halo` tire `Reconstruction`
    from .Reconstruction import Reconstruction

    sino = sinogram if sinogram is not None else mesh.sinogram
    if alphas is None:
        alphas = np.linspace( 0.0, 1.2, 7 )
    alphas = np.asarray( alphas, dtype = float )
    extent = recon_kwargs.pop( "extent", sino.extent )

    out = { k: [] for k in ( "void", "interior_mass", "clipped" ) }
    out.update( { name: [] for name in ( metrics or {} ) } )
    clouds = []
    for a in alphas:
        data = mesh.corrected( sino, alpha = float( a ) )
        rec = Reconstruction( data, extent = extent, **recon_kwargs )
        solve( rec )
        pos = rec.positions
        clouds.append( pos )
        out[ "void" ].append( void_fraction( pos, extent ) )
        out[ "interior_mass" ].append( float( np.asarray( data.mass() ).mean() ) )
        out[ "clipped" ].append( mesh.clipped_fraction( float( a ), sino ) )
        for name, fn in ( metrics or {} ).items():
            out[ name ].append( float( fn( pos ) ) )
        if verbose:
            extra = "".join( f"  { n } { out[ n ][ -1 ]:.4f}" for n in ( metrics or {} ) )
            print( f"[alpha] { a:.3f}: masse intérieure { out[ 'interior_mass' ][ -1 ]:8.1f}  "
                   f"vide { out[ 'void' ][ -1 ]:.1%}  écrêté { out[ 'clipped' ][ -1 ]:.2%}{ extra }" )

    return dict( alphas = alphas, clouds = clouds,
                 **{ k: np.array( v ) for k, v in out.items() } )


class GradedMesh:
    """Maillage de carrés couvrant le disque de rayon `outer_radius`, fin dedans, gradué dehors.

    - `inner_radius` (défaut `extent/2`) : le champ de vue. Les cellules y sont toutes de taille
      `cell_size` -- ce sont elles qu'on remplacera par des diracs (`interior`) ;
    - au-delà, la taille double à chaque niveau, le niveau visé en `r` valant
      `floor( grading · log2( r / inner_radius ) )`. `grading = 2` (défaut) suit la géométrie
      d'acquisition : un point de rayon `r` n'étant vu que sur `2·arcsin( S/r )` de l'acquisition,
      le nombre de mesures qui touchent l'anneau de rayon `r` décroît en `1/r` pendant que sa
      circonférence croît en `r` -- la maille doit croître au moins comme `r²` ;
    - `cell_size` (défaut 4 cases grossières) : la finesse INTÉRIEURE, qui fixe seule le coût. La
      descendre en dessous de la résolution détecteur n'apporte rien -- c'est elle qui borne ce
      que le maillage peut voir.

    Le maillage est un QUADTREE construit par fusion depuis la grille fine : une cellule de niveau
    `L` n'est émise que si les `4^L` cellules fines qu'elle recouvre visent toutes au moins ce
    niveau et sont encore libres. C'est ce qui garantit un vrai PAVAGE -- ni trou ni recouvrement
    aux interfaces, alors qu'une construction par anneaux indépendants en produit forcément (les
    grilles n'y sont pas emboîtées, et la frontière est un cercle). Un opérateur qui compterait
    deux fois une partie du plan fausserait tout le partage de masse, qui est le but de l'étape.

    `nb_coarse_bins` : le maillage travaille sur une grille détecteur regroupée (diviseur exact de
    `nb_bins`). Inutile d'y mettre la pleine résolution : les diracs, eux, la reprendront ensuite
    sur le sinogramme corrigé.

    `weights` (une densité par cellule) démarre à 0 ; `solve` le remplit.
    """

    def __init__( self, sinogram: Sinogram, outer_radius: float, *, inner_radius: float | None = None,
                  cell_size: float | None = None, grading: float = 2.0, max_level: int = 6,
                  nb_coarse_bins: int = 512 ) -> None:
        self.sinogram = sinogram
        self.inner_radius = float( inner_radius if inner_radius is not None else sinogram.extent / 2 )
        self.outer_radius = max( float( outer_radius ), self.inner_radius )
        if grading < 0:
            raise ValueError( "grading doit être >= 0" )

        self.angles = np.asarray( sinogram.angles, dtype = float )
        self.normals = np.asarray( sinogram.normals, dtype = float )
        self.nb_angles = int( self.angles.size )
        self.nb_bins = int( sinogram.nb_bins_host )
        self.s_min = float( sinogram.s_min )
        self.dw = float( sinogram.dw )

        self.group = self._group_size( nb_coarse_bins )
        self.nb_coarse = self.nb_bins // self.group
        self.coarse_dw = self.dw * self.group

        self.cell_size = float( cell_size if cell_size is not None else 4 * self.coarse_dw )
        self.centers, self.sizes, self.levels = self._build_cells( grading, max_level )
        self.areas = self.sizes ** 2
        #: masque des cellules du CHAMP DE VUE -- celles que les diracs remplaceront. Défini par le
        #: rayon du centre, pas par le niveau : une cellule peut rester fine hors du champ (bloc de
        #: fusion incomplet), ce qui n'en fait pas une cellule intérieure.
        self.interior = np.linalg.norm( self.centers, axis = 1 ) < self.inner_radius
        self.weights = np.zeros( self.nb_cells )

        #: les niveaux réellement peuplés (la fusion peut en sauter un), et leurs masques
        self._present = [ ( lv, self.levels == lv ) for lv in np.unique( self.levels ) ]
        self._scatter = self._build_scatter()          # une matrice creuse par niveau présent
        self._kernels = self._build_kernels()          # une FFT de trapèze par niveau
        self._edges = None                             # graphe de la variation totale, à la demande

    def __repr__( self ) -> str:
        per_level = np.bincount( self.levels )
        return ( f"GradedMesh( { self.nb_cells } cellules { list( per_level ) } par niveau, "
                 f"maille { self.cell_size:.3g}, r { self.inner_radius:.3g}..{ self.outer_radius:.3g}, "
                 f"{ self.nb_coarse } cases )" )

    @property
    def nb_cells( self ) -> int:
        return len( self.centers )

    @property
    def nb_levels( self ) -> int:
        return int( self.levels.max() ) + 1

    # -- géométrie ---------------------------------------------------------

    def _group_size( self, nb_coarse_bins: int ) -> int:
        target = max( 1, int( np.ceil( self.nb_bins / max( 1, int( nb_coarse_bins ) ) ) ) )
        for g in range( target, self.nb_bins + 1 ):
            if self.nb_bins % g == 0:
                return g
        return self.nb_bins

    def _build_cells( self, grading, max_level ):
        """Le quadtree, par FUSION depuis la grille fine (cf. la docstring de la classe).

        On part de la grille de pas `cell_size` couvrant le disque, on donne à chaque case un
        niveau VISÉ (croissant avec le rayon), puis on descend les niveaux du plus grossier au plus
        fin : un bloc `2^L x 2^L` aligné sur la grille n'est fusionné que s'il est entièrement dans
        le domaine, entièrement libre, et que toutes ses cases visent au moins `L`. Ce qui reste à
        `L = 0` est émis tel quel, donc chaque case fine appartient à exactement une cellule.
        """
        h = self.cell_size
        # la demi-largeur en cases fines est arrondie à un multiple de 2^max_level, sans quoi les
        # blocs de fusion ne seraient pas alignés sur la grille et la fusion raterait par endroits
        step = 1 << max_level
        n = int( np.ceil( np.ceil( self.outer_radius / h ) / step ) ) * step

        k = np.arange( -n, n )
        cx, cy = np.meshgrid( ( k + 0.5 ) * h, ( k + 0.5 ) * h, indexing = "ij" )
        r = np.hypot( cx, cy )
        free = r < self.outer_radius
        target = np.clip( np.floor( grading * np.log2( np.maximum( r / self.inner_radius, 1.0 ) ) ),
                          0, max_level ).astype( int )

        centers, sizes, levels = [], [], []
        for L in range( max_level, -1, -1 ):
            b = 1 << L
            if L > 0:
                shape = ( 2 * n // b, b, 2 * n // b, b )
                ok = ( free & ( target >= L ) ).reshape( shape ).all( axis = ( 1, 3 ) )
                if not ok.any():
                    continue
                free = free & ~np.repeat( np.repeat( ok, b, axis = 0 ), b, axis = 1 )
                bi, bj = np.nonzero( ok )
                c = np.stack( [ ( ( bi - n // b ) + 0.5 ) * b * h,
                                ( ( bj - n // b ) + 0.5 ) * b * h ], axis = 1 )
            else:
                c = np.stack( [ cx[ free ], cy[ free ] ], axis = 1 )
            centers.append( c )
            sizes.append( np.full( len( c ), b * h ) )
            levels.append( np.full( len( c ), L, dtype = int ) )

        order = np.argsort( np.concatenate( levels ), kind = "stable" )
        return ( np.concatenate( centers )[ order ], np.concatenate( sizes )[ order ],
                 np.concatenate( levels )[ order ] )

    # -- opérateur ---------------------------------------------------------

    def _build_scatter( self ):
        """Par niveau, la matrice creuse `[ nb_angles*nb_coarse, nb_cells_du_niveau ]` qui dépose
        chaque cellule, en masse ponctuelle, à sa position projetée `s = centre.n_θ`.

        Dépôt LINÉAIRE sur les deux cases voisines : la masse est conservée exactement, et l'erreur
        introduite revient à convoluer par un triangle d'une case de large -- une case de flou de
        plus sur un opérateur dont le noyau en fait déjà plusieurs. Ce qui compte davantage ici,
        c'est que l'adjoint soit EXACTEMENT la transposée, ce que la forme matricielle garantit.
        """
        out = []
        for _, mask in self._present:
            c = self.centers[ mask ]
            n = len( c )
            s = self.normals @ c.T                                     # [ nb_angles, n ]
            x = ( s - self.s_min ) / self.coarse_dw - 0.5              # en indices de CENTRES
            j = np.floor( x ).astype( int )
            w = x - j
            base = np.arange( self.nb_angles )[ :, None ] * self.nb_coarse
            col = np.tile( np.arange( n ), ( self.nb_angles, 1 ) )

            rows, cols, vals = [], [], []
            for idx, weight in ( ( j, 1.0 - w ), ( j + 1, w ) ):
                ok = ( idx >= 0 ) & ( idx < self.nb_coarse )
                rows.append( ( base + np.clip( idx, 0, self.nb_coarse - 1 ) )[ ok ] )
                cols.append( col[ ok ] )
                vals.append( weight[ ok ] )
            out.append( sp.csr_matrix(
                ( np.concatenate( vals ), ( np.concatenate( rows ), np.concatenate( cols ) ) ),
                shape = ( self.nb_angles * self.nb_coarse, n ) ) )
        return out

    def _trapezoid( self, h ):
        """Le profil de Radon d'un carré de côté `h`, échantillonné par case, pour chaque angle.

        C'est un TRAPÈZE : avec `a = h|cos θ|`, `b = h|sin θ|`, `u = max( a, b )`, `v = min( a, b )`,
        le support est `|s| < ( u+v )/2`, le plateau `|s| < ( u−v )/2`, et sa hauteur `h²/u` (la
        masse vaut `h²`, l'aire de la cellule -- c'est la vérification du calcul). On intègre sa
        primitive entre bords de case, donc la masse est exacte case par case.

        Renvoie `[ nb_angles, 2*K+1 ]`, en densité, centré : l'indice K est la case du centre.
        """
        a, b = h * np.abs( np.cos( self.angles ) ), h * np.abs( np.sin( self.angles ) )
        u, v = np.maximum( a, b ), np.minimum( a, b )
        p, q = 0.5 * ( u - v ), 0.5 * ( u + v )                        # demi-plateau, demi-support
        H = h * h / np.maximum( u, 1e-300 )

        K = int( np.ceil( q.max() / self.coarse_dw ) ) + 1
        e = ( np.arange( -K, K + 2 ) - 0.5 ) * self.coarse_dw          # bords de case, [ 2K+2 ]
        s = e[ None, : ]
        p, q, H, v = p[ :, None ], q[ :, None ], H[ :, None ], v[ :, None ]

        # primitive du trapèze, écrite par morceaux puis recollée (v = 0 : simple créneau)
        safe_v = np.where( v > 1e-300, v, 1.0 )
        rise = H * np.clip( s + q, 0.0, None ) ** 2 / ( 2 * safe_v )
        fall = H * v / 2 + H * ( s + p )
        top = H * ( p + q ) - H * np.clip( q - s, 0.0, None ) ** 2 / ( 2 * safe_v )   # H(p+q) = h²
        G = np.where( s < -p, np.where( v > 1e-300, rise, 0.0 ),
                      np.where( s < p, fall, top ) )
        G = np.clip( G, 0.0, h * h )
        return ( G[ :, 1: ] - G[ :, :-1 ] ) / self.coarse_dw

    def _build_kernels( self ):
        """Pour chaque niveau, la FFT du trapèze, prête pour la convolution circulaire.

        Le noyau est placé centré-en-0 (décalages négatifs enroulés en fin de tableau) et la
        longueur de FFT dépasse `nb_coarse + longueur du noyau` : ce qui déborderait du détecteur
        atterrit dans la zone de remplissage, qu'on jette. C'est la bonne physique -- la matière
        dont l'ombre sort du capteur n'est tout simplement pas mesurée.
        """
        out = []
        for _, mask in self._present:
            k = self._trapezoid( float( self.sizes[ mask ][ 0 ] ) )
            half = k.shape[ 1 ] // 2
            n = next_fast_len( self.nb_coarse + k.shape[ 1 ] + 1 )
            padded = np.zeros( ( self.nb_angles, n ) )
            padded[ :, : half + 1 ] = k[ :, half: ]                    # décalages 0..+half
            padded[ :, n - half : ] = k[ :, :half ]                    # décalages -half..-1
            out.append( ( rfft( padded, axis = 1 ), n ) )
        return out

    def project( self, weights ) -> np.ndarray:
        """Sinogramme `[ nb_angles, nb_coarse ]` (densité) produit par les densités `weights`."""
        w = np.asarray( weights, dtype = float )
        acc = np.zeros( ( self.nb_angles, self.nb_coarse ) )
        for ( _, mask ), ( kf, n ), scat in zip( self._present, self._kernels, self._scatter ):
            pts = ( scat @ w[ mask ] ).reshape( self.nb_angles, self.nb_coarse )
            acc += irfft( rfft( pts, n = n, axis = 1 ) * kf, n = n, axis = 1 )[ :, :self.nb_coarse ]
        return acc

    def backproject( self, residual ) -> np.ndarray:
        """L'ADJOINT exact de `project` : `[ nb_angles, nb_coarse ]` -> une valeur par cellule.

        Exact et non approché parce que le trapèze est symétrique (la convolution est donc
        auto-adjointe) et que le dépôt est une vraie matrice, qu'on transpose.
        """
        r = np.asarray( residual, dtype = float )
        out = np.zeros( self.nb_cells )
        for ( _, mask ), ( kf, n ), scat in zip( self._present, self._kernels, self._scatter ):
            conv = irfft( rfft( r, n = n, axis = 1 ) * kf, n = n, axis = 1 )[ :, :self.nb_coarse ]
            out[ mask ] = scat.T @ conv.ravel()
        return out

    def lipschitz( self, nb_iter: int = 20, seed: int = 0 ) -> float:
        """`‖AᵀA‖` par la méthode de la puissance -- le pas de FISTA en dépend directement."""
        v = np.random.default_rng( seed ).random( self.nb_cells )
        lam = 1.0
        for _ in range( nb_iter ):
            v = self.backproject( self.project( v ) )
            lam = float( np.linalg.norm( v ) )
            if lam == 0:
                return 1.0
            v /= lam
        return lam

    # -- régularisation ----------------------------------------------------

    def edges( self ):
        """Le graphe de voisinage du maillage, `( i, j, face, face/distance )`, construit une fois.

        Deux cellules sont voisines si la distance de leurs centres est inférieure à
        `0.75·( h_i + h_j )` -- ce qui attrape les voisines de même niveau ET les interfaces
        fin/grossier, sans avoir à traiter la graduation comme un cas particulier.

        Deux poids, parce que les deux régularisations ne mesurent pas la même chose : la longueur
        de FACE `min( h_i, h_j )` fait de `Σ face·|Δ|` un vrai périmètre (variation totale), et
        `face/distance` fait de `Σ ( face/dist )·Δ²` l'énergie de Dirichlet usuelle en volumes
        finis. Sur une zone uniforme les deux valent 1 par arête, donc les deux poids de
        régularisation restent comparables entre eux.
        """
        if self._edges is None:
            tree = cKDTree( self.centers )
            _, idx = tree.query( self.centers, k = min( 9, self.nb_cells ) )
            i = np.repeat( np.arange( self.nb_cells ), idx.shape[ 1 ] )
            j = idx.ravel()
            d = np.linalg.norm( self.centers[ i ] - self.centers[ j ], axis = 1 )
            hi, hj = self.sizes[ i ], self.sizes[ j ]
            keep = ( i < j ) & ( d < 0.75 * ( hi + hj ) )
            i, j, face, d = i[ keep ], j[ keep ], np.minimum( hi, hj )[ keep ], d[ keep ]
            self._edges = ( i, j, face, face / d )
        return self._edges

    def laplacian( self, w ):
        """`L w`, avec `L` le laplacien du graphe pondéré par `face/distance` : le gradient de
        l'énergie de Dirichlet `½ Σ ( face/dist )·( w_i − w_j )²`.

        C'est la régularisation LINÉAIRE. Tout le problème le reste alors -- ni valeur absolue ni
        contrainte -- et se résout par gradient conjugué au lieu de FISTA.
        """
        i, j, _, weight = self.edges()
        g = weight * ( w[ i ] - w[ j ] )
        out = np.zeros_like( w )
        np.add.at( out, i, g )
        np.add.at( out, j, -g )
        return out

    def _tv_grad( self, w, delta ):
        """Gradient de la variation totale LISSÉE (Huber de paramètre `delta`), et sa valeur.

        Le lissage est ce qui permet de rester sur un FISTA ordinaire plutôt que d'écrire un
        primal-dual : au-dessus de `delta` on paie bien `|Δ|` (les fronts sont préservés), en
        dessous on paie `Δ²`, ce qui rend le tout dérivable. `delta` doit rester petit devant les
        sauts de densité qu'on veut garder nets.
        """
        i, j, weight, _ = self.edges()
        d = w[ i ] - w[ j ]
        big = np.abs( d ) > delta
        val = np.where( big, np.abs( d ) - delta / 2, d * d / ( 2 * delta ) )
        g = weight * np.where( big, np.sign( d ), d / delta )
        out = np.zeros_like( w )
        np.add.at( out, i, g )
        np.add.at( out, j, -g )
        return float( weight @ val ), out

    # -- résolution --------------------------------------------------------

    def _target( self, sinogram ):
        """Le sinogramme mesuré, ramené sur la grille grossière du maillage."""
        sino = sinogram if sinogram is not None else self.sinogram
        p = np.asarray( sino.values, dtype = float )
        return p.reshape( self.nb_angles, self.nb_coarse, self.group ).mean( axis = 2 )

    def solve( self, sinogram: Sinogram | None = None, *, smooth: float = 3e-2,
               tv: float | None = None, nb_iter: int | None = None, nonneg: bool = False,
               huber: float | None = None, verbose: bool = False ) -> "GradedMesh":
        """Ajuste `weights` sur le sinogramme mesuré. Met `weights` à jour et renvoie `self`.

        Convexe dans tous les cas -- sans minimum local ni dépendance à l'initialisation, c'est
        tout l'intérêt de passer par un maillage avant les diracs, dont le problème, lui, ne l'est
        pas. Mais deux régimes très différents :

        - `smooth` seul (défaut) : régularisation QUADRATIQUE, problème LINÉAIRE, résolu par
          gradient conjugué sur `( AᵀA + λL ) w = Aᵀp`. Pas de valeur absolue, pas de contrainte,
          pas de pas à estimer -- et une convergence en quelques dizaines d'itérations au lieu de
          plusieurs centaines. C'est le bon choix ici : le maillage n'a pas à résoudre les
          détails intérieurs (c'est le métier des diracs), il doit donner l'empreinte extérieure
          et le partage de masse, deux quantités lisses ;
        - `tv` : variation totale (Huber), qui préserve les fronts mais rend le problème non
          linéaire -- FISTA, un paramètre de lissage de plus, et un coût bien supérieur. À réserver
          au cas où le maillage doit vraiment tenir un bord franc.

        `nonneg` force la positivité, ce qui fait retomber le cas quadratique sur FISTA aussi.

        Les deux poids sont RELATIFS (mis à l'échelle par `‖AᵀA‖`, et par le niveau du sinogramme
        pour la TV) : la même valeur se comporte pareil d'un cas à l'autre.

        NE PAS mettre la régularisation à 0. Mesuré sur `experiments/lung_mesh` (objet deux fois
        plus large que le détecteur), la masse intérieure y semble la MEILLEURE de tout le
        balayage (+5% contre −10%) -- mais la solution est du bruit sel-et-poivre saturé, dont la
        moyenne tombe juste par accident ; la retirer du sinogramme y injecterait ce bruit. Le
        problème intérieur a un noyau non trivial, il FAUT le régulariser, et juger sur la carte,
        pas sur la masse.

        Mesuré sur ce même cas : `smooth = 3e-2` (défaut) donne `M_in` à −12.7% et une masse par
        angle corrigée à 0.25%, contre −10.0% et 0.37% pour `tv = 3e-3`, pour HUIT FOIS moins de
        temps (0.8 s contre 6.8 s). Le quadratique laisse en revanche un anneau parasite au bord du
        champ de vue, que la TV n'a pas : si c'est la CARTE qui compte et non l'empreinte, prendre
        la TV. Essayé et REJETÉ pour supprimer cet anneau : surpondérer les arêtes qui traversent
        le bord du champ (imposer la continuité y dégrade `M_in` de façon monotone, jusqu'à −54%
        pour un facteur 1000).
        """
        p = self._target( sinogram )
        if tv is None and not nonneg:
            return self._solve_cg( p, smooth, 60 if nb_iter is None else int( nb_iter ), verbose )
        return self._solve_fista( p, smooth, tv, 300 if nb_iter is None else int( nb_iter ),
                                  huber, verbose )

    def _solve_cg( self, p, smooth, nb_iter, verbose ):
        """Gradient conjugué sur les équations normales `( AᵀA + λL ) w = Aᵀp` -- le cas linéaire.

        `λ` est calibré pour que la norme de `λL` vaille `smooth·‖AᵀA‖` : le poids est donc sans
        dimension, et directement comparable d'un maillage à l'autre.
        """
        lam = float( smooth ) * self.lipschitz() / max( self._degree().max(), 1e-30 )

        def matvec( v ):
            return self.backproject( self.project( v ) ) + lam * self.laplacian( v )

        b = self.backproject( p )
        w = self.weights.copy()
        r = b - matvec( w )
        d, rr = r.copy(), float( r @ r )
        for it in range( nb_iter ):
            if rr <= 1e-24 * float( b @ b ):
                break
            md = matvec( d )
            alpha = rr / max( float( d @ md ), 1e-300 )
            w += alpha * d
            r -= alpha * md
            rr, rr_old = float( r @ r ), rr
            d = r + ( rr / rr_old ) * d
            if verbose and ( it % 10 == 0 or it == nb_iter - 1 ):
                print( f"  [mesh/cg] it { it:3d}: ‖résidu‖ { np.sqrt( rr ):.4g}"
                       f"  masse { self.mass( w ):.6g} (dont { self.interior_mass( w ):.6g} dedans)" )

        neg = float( -np.minimum( w, 0.0 ).sum() * 1.0 )
        if verbose and neg > 0:
            print( f"  [mesh/cg] négatifs écrêtés : { neg / max( np.abs( w ).sum(), 1e-30 ):.2%} "
                   "de la masse absolue" )
        self.weights = np.maximum( w, 0.0 )
        return self

    def _degree( self ):
        """`Σ_j poids_ij` par cellule, pour le laplacien quadratique -- majore sa norme."""
        i, j, _, weight = self.edges()
        return ( np.bincount( i, weight, self.nb_cells ) + np.bincount( j, weight, self.nb_cells ) )

    def _solve_fista( self, p, smooth, tv, nb_iter, huber, verbose ):
        """FISTA projeté : le cas non linéaire (variation totale et/ou positivité imposée)."""
        lip = self.lipschitz()
        # échelle de densité plausible : la masse mesurée répartie sur le disque du champ de vue
        scale = float( p.sum( axis = 1 ).mean() * self.coarse_dw / ( np.pi * self.inner_radius ** 2 ) )
        delta = float( huber ) if huber is not None else max( 1e-2 * scale, 1e-30 )

        i, j, face, quad = self.edges()
        if tv is None:                                 # quadratique + positivité
            lam, reg_deg = float( smooth ) * lip / max( self._degree().max(), 1e-30 ), self._degree().max()
            reg = lambda w: ( 0.5 * float( quad @ ( w[ i ] - w[ j ] ) ** 2 ), self.laplacian( w ) )
            curvature = lam * 2 * float( reg_deg )
        else:                                          # variation totale (Huber)
            lam = float( tv ) * lip * max( scale, 1e-30 )
            deg = np.bincount( i, face, self.nb_cells ) + np.bincount( j, face, self.nb_cells )
            reg = lambda w: self._tv_grad( w, delta )
            curvature = lam * 2 * float( deg.max() ) / delta
        step = 1.0 / ( lip + curvature )               # majorant de la courbure du terme lissé

        w = self.weights.copy()
        y, t = w.copy(), 1.0
        for it in range( int( nb_iter ) ):
            resid = self.project( y ) - p
            tv_val, tv_grad = reg( y )
            nxt = np.maximum( y - step * ( self.backproject( resid ) + lam * tv_grad ), 0.0 )
            # redémarrage sur le GRADIENT (O'Donoghue-Candès) : quand le pas d'inertie pointe à
            # l'opposé du pas de descente, l'élan travaille contre nous et FISTA se met à osciller
            # -- mesuré ici sur des centaines d'itérations, avec une masse intérieure qui battait
            # de +/-7%. On remet alors l'élan à zéro, ce qui coûte un scalaire par itération.
            if float( ( y - nxt ) @ ( nxt - w ) ) > 0:
                t = 1.0
            t_next = 0.5 * ( 1 + np.sqrt( 1 + 4 * t * t ) )
            y, w, t = nxt + ( ( t - 1 ) / t_next ) * ( nxt - w ), nxt, t_next
            if verbose and ( it % 25 == 0 or it == nb_iter - 1 ):
                print( f"  [mesh/fista] it { it:4d}: ½‖Aw−p‖² { 0.5 * float( ( resid ** 2 ).sum() ):.6g}"
                       f"  régul { tv_val:.6g}  masse { self.mass( w ):.6g}"
                       f" (dont { self.interior_mass( w ):.6g} dedans)" )

        self.weights = w
        return self

    # -- sorties -----------------------------------------------------------

    def mass( self, weights = None ) -> float:
        w = self.weights if weights is None else weights
        return float( np.asarray( w ) @ self.areas )

    def interior_mass( self, weights = None ) -> float:
        """La masse que le maillage attribue au CHAMP DE VUE -- le `M_in` que `halo.py` devait
        deviner, et qu'on lit ici directement sur la solution."""
        w = self.weights if weights is None else weights
        return float( np.asarray( w )[ self.interior ] @ self.areas[ self.interior ] )

    def values( self, mask = None ) -> np.ndarray:
        """Empreinte sur le sinogramme MESURÉ (pleine résolution, `[ nb_angles, nb_bins ]`) des
        cellules retenues par `mask` (toutes par défaut).

        Le retour à la grille fine est une simple répétition : constante par paquet, donc de masse
        exacte -- et l'empreinte varie peu d'une case grossière à l'autre, c'est tout le propos.
        """
        w = self.weights if mask is None else np.where( mask, self.weights, 0.0 )
        return np.repeat( self.project( w ), self.group, axis = 1 )

    def exterior_values( self, alpha: float = 1.0 ) -> np.ndarray:
        """L'empreinte des seules cellules HORS champ de vue, multipliée par `alpha` -- ce qu'il
        faut retirer du sinogramme avant de rendre l'intérieur aux diracs.

        `alpha` est le SEUL degré de liberté qui reste vraiment ouvert (cf. `scan_exterior_scale`) :
        le noyau du problème intérieur permet de déplacer un niveau lentement variable entre le
        dedans et le dehors, et la régularisation en choisit un arbitrairement.
        """
        return float( alpha ) * self.values( ~self.interior )

    def corrected( self, sinogram: Sinogram | None = None, alpha: float = 1.0 ) -> Sinogram:
        """Le sinogramme débarrassé de la contribution extérieure, écrêté à 0."""
        sino = sinogram if sinogram is not None else self.sinogram
        out = Sinogram( nb_angles = self.nb_angles, nb_bins = self.nb_bins,
                        extent = sino.extent, detector_center = sino.detector_center )
        out.values = np.clip( np.asarray( sino.values ) - self.exterior_values( alpha ), 0.0, None )
        return out

    def clipped_fraction( self, alpha: float = 1.0, sinogram: Sinogram | None = None ) -> float:
        """Part de masse que l'écrêtage à 0 doit inventer, `Σ max( 0, α·E − p ) / Σ p`.

        Indicateur MODÈLE-LIBRE de sur-soustraction : la projection de l'intérieur étant positive,
        `α·E` ne peut pas dépasser `p`. Tant qu'il reste nul, `alpha` est physiquement admissible ;
        il ne dit en revanche rien de la SOUS-soustraction, et reste donc muet quand l'objet remplit
        le détecteur à tous les angles.
        """
        sino = sinogram if sinogram is not None else self.sinogram
        p = np.asarray( sino.values, dtype = float )
        return float( np.clip( self.exterior_values( alpha ) - p, 0.0, None ).sum() / max( p.sum(), 1e-30 ) )
