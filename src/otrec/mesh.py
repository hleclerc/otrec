"""Reconstruction on a graded MESH covering the whole object, detector included and overflowed.

This is the step that precedes the diracs when the part is wider than the detector. The
reasoning, in three stages:

1. we first solve EVERYTHING on a mesh -- the interior of the field of view in fine cells,
   the exterior in coarser and coarser cells. A CONVEX problem, so no local minimum,
   no initialization to take care of;
2. the mesh then gives the interior/exterior split of the mass, which was NOT accessible
   otherwise. This settles the hard point of `halo.py`: there `M_in` had to be guessed
   (bounded by `min_θ ∫p_θ`, a bound that is off by 50% as soon as no angle sees the whole object), here it
   is read off the solution;
3. we REMOVE from the sinogram the contribution of the EXTERIOR cells, and hand the interior back to the
   diracs/disks, which is their job -- only they know how to leave voids. The mesh
   is too smooth for that: it is a background estimator, not a structure estimator.

The grading comes from the acquisition geometry: a point of radius r is only seen over the
angular window `2·arcsin( S/r )`, so the available information collapses with distance. Fine
cells out there would only absorb interior signal, and the regularization takes over.

It is QUADRATIC by default (see `solve`), so the problem stays LINEAR and is solved by
conjugate gradient. This is the right compromise here because the mesh does not have to resolve the interior
details -- that is the diracs' job: it must deliver the exterior footprint and the mass
split, two smooth quantities. Total variation remains available, and gives a cleaner map
at the edge of the field of view, but makes the problem nonlinear for eight times the computation time.

== How the operator is computed (and why there is no matrix)

All cells are SQUARES, only their size changes. Now the projection of a square of side
h is an analytic trapezoid that only depends on `( h, θ )` -- not on the position, which merely
SHIFTS it. The projection of a mesh level is therefore:

    project = deposit the weights at the projected positions of the centers, then CONVOLVE by the trapezoid

The deposit is a fixed sparse matrix (two bins per cell and per angle, linear deposit), the
convolution an FFT per angle with a precomputed kernel. No system matrix: memory is
`O( nb_cells x nb_angles )` instead of `O( nb_cells x nb_angles x nb_bins )`. The trapezoid
being SYMMETRIC, the convolution is self-adjoint and the adjoint of the full operator is
exactly the transposed deposit -- no approximate adjoint, hence a solver that truly converges.
"""
import numpy as np
import scipy.sparse as sp
from scipy.fft import irfft, next_fast_len, rfft
from scipy.spatial import cKDTree

from .Sinogram import Sinogram


def scan_exterior_scale( mesh, solve, alphas = None, *, sinogram: Sinogram | None = None,
                         metrics = None, verbose: bool = False, **recon_kwargs ) -> dict:
    """Sweeps `alpha`, the factor applied to the exterior footprint before subtraction, and
    reconstructs with diracs for each.

    Why a sweep rather than a computation: the kernel of the interior problem allows moving
    a slowly varying level between the inside and the outside, and nothing in the data says
    which one to choose -- that is the definition of a kernel. Any regularization picks one, and we have
    measured (see `solve`) that this choice is worth about 10% of the interior mass. Rather than
    hiding it in a regularization weight, we expose it as ONE scalar, which can be confronted with
    what we have available elsewhere.

    `solve( rec )`: one complete interior solve, as for `halo.alternate`.
    `metrics`: `{ name: f( positions ) -> float }` in addition to those computed by default
    (`void_fraction`, the remaining mass per angle, and `clipped_fraction`, which tells from which
    `alpha` the subtraction becomes physically impossible).

    Returns a dict of arrays plus `clouds`, the list of the clouds obtained.

    MEASURED, and this is a warning: on `experiments/lung_mesh`, the true optimum is clear
    (minimal density error at α = 0.90, where the interior mass falls back to 1225 versus 1226)
    but NO observable indicator finds it. The void grows monotonically (33% to 43% between
    α = 0.9 and 1.1), the clipping stays zero up to α = 1, and the OT residual only grows. This is
    the very definition of a kernel: nothing in the data distinguishes these solutions. This sweep
    therefore serves to EXPLOIT an exterior anchor (known material density, tighter support,
    a few wide-field views), not to do without one. Consolation: the error is flat around
    the optimum -- 0.450 at α = 0.90 against 0.471 at α = 1 --, so the default α = 1 remains reasonable.
    Only the OT residual shows the beginnings of an elbow towards the optimum; to be confronted with other cases
    before making it a criterion.
    """
    from .halo import void_fraction                    # late: `halo` pulls in `Reconstruction`
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
            print( f"[alpha] { a:.3f}: interior mass { out[ 'interior_mass' ][ -1 ]:8.1f}  "
                   f"void { out[ 'void' ][ -1 ]:.1%}  clipped { out[ 'clipped' ][ -1 ]:.2%}{ extra }" )

    return dict( alphas = alphas, clouds = clouds,
                 **{ k: np.array( v ) for k, v in out.items() } )


class GradedMesh:
    """Mesh of squares covering the disk of radius `outer_radius`, fine inside, graded outside.

    - `inner_radius` (default `extent/2`): the field of view. The cells there are all of size
      `cell_size` -- these are the ones that will be replaced by diracs (`interior`);
    - beyond, the size doubles at each level, the target level at `r` being
      `floor( grading · log2( r / inner_radius ) )`. `grading = 2` (default) follows the acquisition
      geometry: a point of radius `r` being seen over only `2·arcsin( S/r )` of the acquisition,
      the number of measurements touching the ring of radius `r` decreases as `1/r` while its
      circumference grows as `r` -- the mesh size must grow at least like `r²`;
    - `cell_size` (default 4 coarse bins): the INTERIOR fineness, which alone sets the cost.
      Going below the detector resolution brings nothing -- it is what bounds
      what the mesh can see.

    The mesh is a QUADTREE built by merging from the fine grid: a cell of level
    `L` is only emitted if the `4^L` fine cells it covers all target at least that
    level and are still free. This is what guarantees a true TILING -- no gap or overlap
    at the interfaces, whereas a construction by independent rings necessarily produces them (the
    grids are not nested there, and the boundary is a circle). An operator that counted
    part of the plane twice would distort the whole mass split, which is the goal of this step.

    `nb_coarse_bins`: the mesh works on a grouped detector grid (exact divisor of
    `nb_bins`). No point in putting the full resolution there: the diracs will pick it up afterwards
    on the corrected sinogram.

    `weights` (one density per cell) starts at 0; `solve` fills it.
    """

    def __init__( self, sinogram: Sinogram, outer_radius: float, *, inner_radius: float | None = None,
                  cell_size: float | None = None, grading: float = 2.0, max_level: int = 6,
                  nb_coarse_bins: int = 512 ) -> None:
        self.sinogram = sinogram
        self.inner_radius = float( inner_radius if inner_radius is not None else sinogram.extent / 2 )
        self.outer_radius = max( float( outer_radius ), self.inner_radius )
        if grading < 0:
            raise ValueError( "grading must be >= 0" )

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
        #: mask of the FIELD OF VIEW cells -- those the diracs will replace. Defined by the
        #: radius of the center, not by the level: a cell may remain fine outside the field (incomplete
        #: merge block), which does not make it an interior cell.
        self.interior = np.linalg.norm( self.centers, axis = 1 ) < self.inner_radius
        self.weights = np.zeros( self.nb_cells )

        #: the levels actually populated (the merge may skip one), and their masks
        self._present = [ ( lv, self.levels == lv ) for lv in np.unique( self.levels ) ]
        self._scatter = self._build_scatter()          # one sparse matrix per present level
        self._kernels = self._build_kernels()          # one trapezoid FFT per level
        self._edges = None                             # total variation graph, on demand

    def __repr__( self ) -> str:
        per_level = np.bincount( self.levels )
        return ( f"GradedMesh( { self.nb_cells } cells { list( per_level ) } per level, "
                 f"size { self.cell_size:.3g}, r { self.inner_radius:.3g}..{ self.outer_radius:.3g}, "
                 f"{ self.nb_coarse } bins )" )

    @property
    def nb_cells( self ) -> int:
        return len( self.centers )

    @property
    def nb_levels( self ) -> int:
        return int( self.levels.max() ) + 1

    # -- geometry ----------------------------------------------------------

    def _group_size( self, nb_coarse_bins: int ) -> int:
        target = max( 1, int( np.ceil( self.nb_bins / max( 1, int( nb_coarse_bins ) ) ) ) )
        for g in range( target, self.nb_bins + 1 ):
            if self.nb_bins % g == 0:
                return g
        return self.nb_bins

    def _build_cells( self, grading, max_level ):
        """The quadtree, by MERGING from the fine grid (cf. the class docstring).

        We start from the grid of step `cell_size` covering the disk, give each cell a
        TARGET level (increasing with the radius), then go down the levels from coarsest to
        finest: a `2^L x 2^L` block aligned on the grid is only merged if it is entirely in
        the domain, entirely free, and all its cells target at least `L`. What remains at
        `L = 0` is emitted as is, so each fine cell belongs to exactly one cell.
        """
        h = self.cell_size
        # the half-width in fine cells is rounded to a multiple of 2^max_level, otherwise the
        # merge blocks would not be aligned on the grid and the merge would miss in places
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

    # -- operator ----------------------------------------------------------

    def _build_scatter( self ):
        """Per level, the sparse matrix `[ nb_angles*nb_coarse, nb_cells_of_the_level ]` that deposits
        each cell, as a point mass, at its projected position `s = center.n_θ`.

        LINEAR deposit on the two neighboring bins: the mass is conserved exactly, and the error
        introduced amounts to convolving by a triangle one bin wide -- one more bin of blur
        on an operator whose kernel already has several. What matters more here
        is that the adjoint be EXACTLY the transpose, which the matrix form guarantees.
        """
        out = []
        for _, mask in self._present:
            c = self.centers[ mask ]
            n = len( c )
            s = self.normals @ c.T                                     # [ nb_angles, n ]
            x = ( s - self.s_min ) / self.coarse_dw - 0.5              # in CENTER indices
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
        """The Radon profile of a square of side `h`, sampled per bin, for each angle.

        It is a TRAPEZOID: with `a = h|cos θ|`, `b = h|sin θ|`, `u = max( a, b )`, `v = min( a, b )`,
        the support is `|s| < ( u+v )/2`, the plateau `|s| < ( u−v )/2`, and its height `h²/u` (the
        mass is `h²`, the area of the cell -- this is the check of the computation). We integrate its
        primitive between bin edges, so the mass is exact bin by bin.

        Returns `[ nb_angles, 2*K+1 ]`, as a density, centered: index K is the center bin.
        """
        a, b = h * np.abs( np.cos( self.angles ) ), h * np.abs( np.sin( self.angles ) )
        u, v = np.maximum( a, b ), np.minimum( a, b )
        p, q = 0.5 * ( u - v ), 0.5 * ( u + v )                        # half-plateau, half-support
        H = h * h / np.maximum( u, 1e-300 )

        K = int( np.ceil( q.max() / self.coarse_dw ) ) + 1
        e = ( np.arange( -K, K + 2 ) - 0.5 ) * self.coarse_dw          # bin edges, [ 2K+2 ]
        s = e[ None, : ]
        p, q, H, v = p[ :, None ], q[ :, None ], H[ :, None ], v[ :, None ]

        # primitive of the trapezoid, written piecewise then glued back (v = 0: simple box)
        safe_v = np.where( v > 1e-300, v, 1.0 )
        rise = H * np.clip( s + q, 0.0, None ) ** 2 / ( 2 * safe_v )
        fall = H * v / 2 + H * ( s + p )
        top = H * ( p + q ) - H * np.clip( q - s, 0.0, None ) ** 2 / ( 2 * safe_v )   # H(p+q) = h²
        G = np.where( s < -p, np.where( v > 1e-300, rise, 0.0 ),
                      np.where( s < p, fall, top ) )
        G = np.clip( G, 0.0, h * h )
        return ( G[ :, 1: ] - G[ :, :-1 ] ) / self.coarse_dw

    def _build_kernels( self ):
        """For each level, the FFT of the trapezoid, ready for circular convolution.

        The kernel is placed centered-at-0 (negative shifts wrapped to the end of the array) and the
        FFT length exceeds `nb_coarse + kernel length`: whatever would overflow the detector
        lands in the padding zone, which is discarded. This is the right physics -- matter
        whose shadow falls off the sensor is simply not measured.
        """
        out = []
        for _, mask in self._present:
            k = self._trapezoid( float( self.sizes[ mask ][ 0 ] ) )
            half = k.shape[ 1 ] // 2
            n = next_fast_len( self.nb_coarse + k.shape[ 1 ] + 1 )
            padded = np.zeros( ( self.nb_angles, n ) )
            padded[ :, : half + 1 ] = k[ :, half: ]                    # shifts 0..+half
            padded[ :, n - half : ] = k[ :, :half ]                    # shifts -half..-1
            out.append( ( rfft( padded, axis = 1 ), n ) )
        return out

    def project( self, weights ) -> np.ndarray:
        """Sinogram `[ nb_angles, nb_coarse ]` (density) produced by the densities `weights`."""
        w = np.asarray( weights, dtype = float )
        acc = np.zeros( ( self.nb_angles, self.nb_coarse ) )
        for ( _, mask ), ( kf, n ), scat in zip( self._present, self._kernels, self._scatter ):
            pts = ( scat @ w[ mask ] ).reshape( self.nb_angles, self.nb_coarse )
            acc += irfft( rfft( pts, n = n, axis = 1 ) * kf, n = n, axis = 1 )[ :, :self.nb_coarse ]
        return acc

    def backproject( self, residual ) -> np.ndarray:
        """The exact ADJOINT of `project`: `[ nb_angles, nb_coarse ]` -> one value per cell.

        Exact and not approximate because the trapezoid is symmetric (the convolution is thus
        self-adjoint) and the deposit is a true matrix, which we transpose.
        """
        r = np.asarray( residual, dtype = float )
        out = np.zeros( self.nb_cells )
        for ( _, mask ), ( kf, n ), scat in zip( self._present, self._kernels, self._scatter ):
            conv = irfft( rfft( r, n = n, axis = 1 ) * kf, n = n, axis = 1 )[ :, :self.nb_coarse ]
            out[ mask ] = scat.T @ conv.ravel()
        return out

    def lipschitz( self, nb_iter: int = 20, seed: int = 0 ) -> float:
        """`‖AᵀA‖` by the power method -- the FISTA step depends directly on it."""
        v = np.random.default_rng( seed ).random( self.nb_cells )
        lam = 1.0
        for _ in range( nb_iter ):
            v = self.backproject( self.project( v ) )
            lam = float( np.linalg.norm( v ) )
            if lam == 0:
                return 1.0
            v /= lam
        return lam

    # -- regularization ----------------------------------------------------

    def edges( self ):
        """The neighborhood graph of the mesh, `( i, j, face, face/distance )`, built once.

        Two cells are neighbors if the distance between their centers is less than
        `0.75·( h_i + h_j )` -- which catches same-level neighbors AND fine/coarse
        interfaces, without having to treat the grading as a special case.

        Two weights, because the two regularizations do not measure the same thing: the
        FACE length `min( h_i, h_j )` makes `Σ face·|Δ|` a true perimeter (total variation), and
        `face/distance` makes `Σ ( face/dist )·Δ²` the usual finite-volume Dirichlet energy. On a
        uniform zone both equal 1 per edge, so the two regularization weights
        remain comparable with each other.
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
        """`L w`, with `L` the Laplacian of the graph weighted by `face/distance`: the gradient of
        the Dirichlet energy `½ Σ ( face/dist )·( w_i − w_j )²`.

        This is the LINEAR regularization. The whole problem then stays so -- no absolute value nor
        constraint -- and is solved by conjugate gradient instead of FISTA.
        """
        i, j, _, weight = self.edges()
        g = weight * ( w[ i ] - w[ j ] )
        out = np.zeros_like( w )
        np.add.at( out, i, g )
        np.add.at( out, j, -g )
        return out

    def _tv_grad( self, w, delta ):
        """Gradient of the SMOOTHED total variation (Huber with parameter `delta`), and its value.

        The smoothing is what lets us stay on an ordinary FISTA rather than writing a
        primal-dual: above `delta` we do pay `|Δ|` (fronts are preserved), below
        we pay `Δ²`, which makes the whole differentiable. `delta` must stay small compared to the
        density jumps we want to keep sharp.
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

    # -- solving -----------------------------------------------------------

    def _target( self, sinogram ):
        """The measured sinogram, brought back onto the coarse grid of the mesh."""
        sino = sinogram if sinogram is not None else self.sinogram
        p = np.asarray( sino.values, dtype = float )
        return p.reshape( self.nb_angles, self.nb_coarse, self.group ).mean( axis = 2 )

    def solve( self, sinogram: Sinogram | None = None, *, smooth: float = 3e-2,
               tv: float | None = None, nb_iter: int | None = None, nonneg: bool = False,
               huber: float | None = None, verbose: bool = False ) -> "GradedMesh":
        """Fits `weights` to the measured sinogram. Updates `weights` and returns `self`.

        Convex in all cases -- with no local minimum nor dependence on initialization, which is
        the whole point of going through a mesh before the diracs, whose problem is not. But two very
        different regimes:

        - `smooth` alone (default): QUADRATIC regularization, LINEAR problem, solved by
          conjugate gradient on `( AᵀA + λL ) w = Aᵀp`. No absolute value, no constraint,
          no step to estimate -- and convergence in a few dozen iterations instead of
          several hundred. This is the right choice here: the mesh does not have to resolve the interior
          details (that is the diracs' job), it must give the exterior footprint
          and the mass split, two smooth quantities;
        - `tv`: total variation (Huber), which preserves fronts but makes the problem
          nonlinear -- FISTA, one more smoothing parameter, and a much higher cost. To be reserved
          for the case where the mesh must really hold a sharp edge.

        `nonneg` forces positivity, which makes the quadratic case fall back on FISTA too.

        The two weights are RELATIVE (scaled by `‖AᵀA‖`, and by the sinogram level
        for the TV): the same value behaves the same from one case to another.

        Do NOT set the regularization to 0. Measured on `experiments/lung_mesh` (object twice
        as wide as the detector), the interior mass there seems the BEST of the whole
        sweep (+5% against −10%) -- but the solution is saturated salt-and-pepper noise, whose
        mean comes out right by accident; removing it from the sinogram would inject that noise there. The
        interior problem has a non-trivial kernel, it MUST be regularized, and judged on the map,
        not on the mass.

        Measured on this same case: `smooth = 3e-2` (default) gives `M_in` at −12.7% and a corrected
        mass per angle at 0.25%, against −10.0% and 0.37% for `tv = 3e-3`, for EIGHT TIMES less
        time (0.8 s versus 6.8 s). The quadratic does leave a parasitic ring at the edge of the
        field of view, which the TV does not: if it is the MAP that matters and not the footprint, take
        the TV. Tried and REJECTED to suppress this ring: overweighting the edges that cross
        the field boundary (imposing continuity there degrades `M_in` monotonically, down to −54%
        for a factor 1000).
        """
        p = self._target( sinogram )
        if tv is None and not nonneg:
            return self._solve_cg( p, smooth, 60 if nb_iter is None else int( nb_iter ), verbose )
        return self._solve_fista( p, smooth, tv, 300 if nb_iter is None else int( nb_iter ),
                                  huber, verbose )

    def _solve_cg( self, p, smooth, nb_iter, verbose ):
        """Conjugate gradient on the normal equations `( AᵀA + λL ) w = Aᵀp` -- the linear case.

        `λ` is calibrated so that the norm of `λL` equals `smooth·‖AᵀA‖`: the weight is thus
        dimensionless, and directly comparable from one mesh to another.
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
                print( f"  [mesh/cg] it { it:3d}: ‖residual‖ { np.sqrt( rr ):.4g}"
                       f"  mass { self.mass( w ):.6g} (of which { self.interior_mass( w ):.6g} inside)" )

        neg = float( -np.minimum( w, 0.0 ).sum() * 1.0 )
        if verbose and neg > 0:
            print( f"  [mesh/cg] negatives clipped: { neg / max( np.abs( w ).sum(), 1e-30 ):.2%} "
                   "of the absolute mass" )
        self.weights = np.maximum( w, 0.0 )
        return self

    def _degree( self ):
        """`Σ_j weight_ij` per cell, for the quadratic Laplacian -- bounds its norm."""
        i, j, _, weight = self.edges()
        return ( np.bincount( i, weight, self.nb_cells ) + np.bincount( j, weight, self.nb_cells ) )

    def _solve_fista( self, p, smooth, tv, nb_iter, huber, verbose ):
        """Projected FISTA: the nonlinear case (total variation and/or imposed positivity)."""
        lip = self.lipschitz()
        # plausible density scale: the measured mass spread over the disk of the field of view
        scale = float( p.sum( axis = 1 ).mean() * self.coarse_dw / ( np.pi * self.inner_radius ** 2 ) )
        delta = float( huber ) if huber is not None else max( 1e-2 * scale, 1e-30 )

        i, j, face, quad = self.edges()
        if tv is None:                                 # quadratic + positivity
            lam, reg_deg = float( smooth ) * lip / max( self._degree().max(), 1e-30 ), self._degree().max()
            reg = lambda w: ( 0.5 * float( quad @ ( w[ i ] - w[ j ] ) ** 2 ), self.laplacian( w ) )
            curvature = lam * 2 * float( reg_deg )
        else:                                          # total variation (Huber)
            lam = float( tv ) * lip * max( scale, 1e-30 )
            deg = np.bincount( i, face, self.nb_cells ) + np.bincount( j, face, self.nb_cells )
            reg = lambda w: self._tv_grad( w, delta )
            curvature = lam * 2 * float( deg.max() ) / delta
        step = 1.0 / ( lip + curvature )               # upper bound of the curvature of the smooth term

        w = self.weights.copy()
        y, t = w.copy(), 1.0
        for it in range( int( nb_iter ) ):
            resid = self.project( y ) - p
            tv_val, tv_grad = reg( y )
            nxt = np.maximum( y - step * ( self.backproject( resid ) + lam * tv_grad ), 0.0 )
            # restart on the GRADIENT (O'Donoghue-Candès): when the momentum step points
            # opposite to the descent step, the momentum works against us and FISTA starts to oscillate
            # -- measured here over hundreds of iterations, with an interior mass that fluctuated
            # by +/-7%. We then reset the momentum to zero, which costs one scalar per iteration.
            if float( ( y - nxt ) @ ( nxt - w ) ) > 0:
                t = 1.0
            t_next = 0.5 * ( 1 + np.sqrt( 1 + 4 * t * t ) )
            y, w, t = nxt + ( ( t - 1 ) / t_next ) * ( nxt - w ), nxt, t_next
            if verbose and ( it % 25 == 0 or it == nb_iter - 1 ):
                print( f"  [mesh/fista] it { it:4d}: ½‖Aw−p‖² { 0.5 * float( ( resid ** 2 ).sum() ):.6g}"
                       f"  regul { tv_val:.6g}  mass { self.mass( w ):.6g}"
                       f" (of which { self.interior_mass( w ):.6g} inside)" )

        self.weights = w
        return self

    # -- outputs -----------------------------------------------------------

    def mass( self, weights = None ) -> float:
        w = self.weights if weights is None else weights
        return float( np.asarray( w ) @ self.areas )

    def interior_mass( self, weights = None ) -> float:
        """The mass the mesh attributes to the FIELD OF VIEW -- the `M_in` that `halo.py` had to
        guess, and which is read here directly off the solution."""
        w = self.weights if weights is None else weights
        return float( np.asarray( w )[ self.interior ] @ self.areas[ self.interior ] )

    def values( self, mask = None ) -> np.ndarray:
        """Footprint on the MEASURED sinogram (full resolution, `[ nb_angles, nb_bins ]`) of the
        cells selected by `mask` (all by default).

        The return to the fine grid is a simple repetition: constant per packet, hence of exact
        mass -- and the footprint varies little from one coarse bin to the next, which is the whole point.
        """
        w = self.weights if mask is None else np.where( mask, self.weights, 0.0 )
        return np.repeat( self.project( w ), self.group, axis = 1 )

    def exterior_values( self, alpha: float = 1.0 ) -> np.ndarray:
        """The footprint of only the cells OUTSIDE the field of view, multiplied by `alpha` -- what
        must be removed from the sinogram before handing the interior back to the diracs.

        `alpha` is the ONLY degree of freedom that remains truly open (cf. `scan_exterior_scale`):
        the kernel of the interior problem allows moving a slowly varying level between the
        inside and the outside, and the regularization picks one arbitrarily.
        """
        return float( alpha ) * self.values( ~self.interior )

    def corrected( self, sinogram: Sinogram | None = None, alpha: float = 1.0 ) -> Sinogram:
        """The sinogram stripped of the exterior contribution, clipped at 0."""
        sino = sinogram if sinogram is not None else self.sinogram
        out = Sinogram( nb_angles = self.nb_angles, nb_bins = self.nb_bins,
                        extent = sino.extent, detector_center = sino.detector_center )
        out.values = np.clip( np.asarray( sino.values ) - self.exterior_values( alpha ), 0.0, None )
        return out

    def clipped_fraction( self, alpha: float = 1.0, sinogram: Sinogram | None = None ) -> float:
        """Share of mass that the clipping at 0 has to invent, `Σ max( 0, α·E − p ) / Σ p`.

        MODEL-FREE indicator of over-subtraction: the projection of the interior being positive,
        `α·E` cannot exceed `p`. As long as it stays zero, `alpha` is physically admissible;
        it says nothing, however, about UNDER-subtraction, and thus stays silent when the object fills
        the detector at all angles.
        """
        sino = sinogram if sinogram is not None else self.sinogram
        p = np.asarray( sino.values, dtype = float )
        return float( np.clip( self.exterior_values( alpha ) - p, 0.0, None ).sum() / max( p.sum(), 1e-30 ) )
