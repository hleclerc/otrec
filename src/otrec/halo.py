"""The HALO: the matter OUTSIDE the field of view, and its footprint on the sinogram.

When the observed part is wider than the detector, each measured profile contains the
contribution of matter that is NOT reconstructible: at angle θ, a point of radius r > S
(S = extent/2) is only seen over the angular window `2·arcsin( S/r )`. The measured mass
`∫p_θ` therefore varies with the angle, whereas `SdotPlan1d` normalizes both distributions to mass 1:
the excess is redistributed INSIDE the field, and fills in the voids that the diracs/disks
were precisely there to preserve.

This module removes that excess. The design choice, which explains everything that follows:

    we do NOT try to reconstruct the exterior, only its FOOTPRINT `c( θ, s )`.

The data cover all lines hitting the disk of radius S -- this is the *interior problem*
of the Radon transform, whose kernel is non-trivial: the exterior is not identifiable in
detail. Meshing it finely would create degrees of freedom that would go and
absorb real interior signal. Two design consequences:

- the halo is COARSE, and gets coarser and coarser towards the outside (see `_build_cells`);
- it is not a free function of `( θ, s )` either -- that would be too many DOFs, and degenerate
  with the interior. It is constrained to be the Radon transform of a density ≥ 0 supported
  outside the disk of radius S, which FOR FREE enforces angular consistency (Helgason-Ludwig
  moment conditions). It is this constraint that makes the estimation well posed with very
  few parameters.

What makes the interior/exterior split identifiable in PRACTICE although it is not so in
the operator sense: the two priors are orthogonal in scale -- the interior is sparse
and wants to leave voids, the halo is smooth and coarse.

`Halo` is FIXED (the mesh does not move, unlike the diracs): the map
`cell densities -> contribution to the sinogram` is therefore a LINEAR operator precomputed
once, and the estimation is a POSITIVE least squares -- convex, with no non-convexity
added to the problem.

`alternate` chains the two blocks (the halo estimation needs the interior, and
vice versa); two or three passes are enough, precisely because they live at different
scales.

`Sinogram.debias_and_equalize_mass` is the degenerate case of this module: a halo with ONE degree of
freedom per angle (`c_k` constant in `s`), with no geometric link between angles. Correct
when the object is MUCH larger than the window, wrong as soon as it only exceeds it by a factor
1.5-2 -- the troublesome case.
"""
import numpy as np
from scipy.optimize import nnls

from .disks import DiskProjector
from .Reconstruction import Reconstruction
from .Sinogram import Sinogram


class Halo:
    """FIXED log-polar mesh of the exterior of the field of view, its projection operator, and
    the fitted densities.

    Geometry (everything is in WORLD coordinates, those of `Sinogram`):
    - `inner_radius` (default `extent/2`): the edge of the field of view -- none of the halo enters it,
      it is the domain reserved for the diracs/disks;
    - `outer_radius`: how far the matter extends. Better to overestimate than underestimate: a useless
      cell gets weight 0 (positivity takes care of it), a missing cell cannot be
      invented;
    - `growth`: ratio of the radii of two consecutive rings (geometrically
      growing thickness);
    - `nb_sectors`: number of sectors a ring located EXACTLY at `inner_radius` would have;
      the real rings have fewer, more and more so as they move away (see `_build_cells`). The
      default of 32 is the elbow measured on a compact exterior object (the most demanding case):
      below it, the mesh no longer localizes the object and the visible mass per angle drops off;
      above it, the error no longer decreases -- it is then limited by the coarse grid.

    `nb_coarse_bins`: the halo is estimated on a COARSE detector grid (the measured bins
    are grouped there in whole packets). This is not just a saving: it is one more
    regularization, and it is consistent with the premise -- the halo footprint has no fine structure.

    The weights (`weights`, one density per cell) start at 0 -- a new halo corrects nothing.
    `fit` updates them; all output methods (`values`, `corrected`, `mass`) read them.
    """

    def __init__( self, sinogram: Sinogram, outer_radius: float, *, inner_radius: float | None = None,
                  growth: float = 1.6, nb_sectors: int = 32, nb_coarse_bins: int = 256,
                  phi_per_bin: float = 4.0, nb_phi_max: int = 1 << 15 ) -> None:
        self.sinogram = sinogram
        self.inner_radius = float( inner_radius if inner_radius is not None else sinogram.extent / 2 )
        self.outer_radius = float( outer_radius )
        if self.outer_radius <= self.inner_radius:
            raise ValueError( f"outer_radius ({ self.outer_radius }) must exceed inner_radius ({ self.inner_radius })" )
        if growth <= 1.0:
            raise ValueError( "growth must be > 1" )
        self.growth = float( growth )
        self.nb_sectors = max( 1, int( nb_sectors ) )
        self.phi_per_bin = float( phi_per_bin )
        self.nb_phi_max = max( 8, int( nb_phi_max ) )

        # detector geometry, HOST side (this whole module is numpy: neither differentiation nor jit --
        # the halo is a CONSTANT from the point of view of the interior optimization)
        self.angles = np.asarray( sinogram.angles, dtype = float )
        self.nb_angles = int( self.angles.size )
        self.nb_bins = int( sinogram.nb_bins_host )
        self.s_min = float( sinogram.s_min )
        self.dw = float( sinogram.dw )

        # grouping of the measured bins into coarse bins -- an exact divisor of `nb_bins`,
        # so that the grouping (mean) and its inverse (`np.repeat`) conserve mass.
        self.group = self._group_size( nb_coarse_bins )
        self.nb_coarse = self.nb_bins // self.group
        self.coarse_dw = self.dw * self.group
        self.coarse_edges = self.s_min + self.coarse_dw * np.arange( self.nb_coarse + 1 )

        self.cells = self._build_cells()                                   # [ ( a, b, phi0, phi1, ring ) ]
        #: `[ nb_cells, nb_angles, nb_coarse ]` -- projected density of a cell of density 1
        self.operator = np.stack( [ self._cell_values( *c[ :4 ] ) for c in self.cells ] )
        #: area of each cell, `[ nb_cells ]` (mass of a cell = area x density)
        self.areas = np.array( [ 0.5 * ( b * b - a * a ) * ( p1 - p0 ) for a, b, p0, p1, _ in self.cells ] )
        #: fitted density per cell, `[ nb_cells ]` -- 0 until `fit` has run
        self.weights = np.zeros( len( self.cells ) )

    def __repr__( self ) -> str:
        return ( f"Halo( { self.nb_cells } cells, r { self.inner_radius:.3g}..{ self.outer_radius:.3g}, "
                 f"{ self.nb_coarse } coarse bins )" )

    @property
    def nb_cells( self ) -> int:
        return len( self.cells )

    # -- geometry ----------------------------------------------------------

    def _group_size( self, nb_coarse_bins: int ) -> int:
        """The smallest divisor of `nb_bins` giving at most `nb_coarse_bins` coarse bins."""
        target = max( 1, int( np.ceil( self.nb_bins / max( 1, int( nb_coarse_bins ) ) ) ) )
        for g in range( target, self.nb_bins + 1 ):
            if self.nb_bins % g == 0:
                return g
        return self.nb_bins

    def _build_cells( self ):
        """Rings of geometric thickness, cut into fewer and fewer sectors.

        The decay law comes from the acquisition geometry: a point of radius r is only seen
        over an angular window `2·arcsin( S/r ) ≈ 2S/r`, so the number of measurements touching
        the ring of radius r decreases as `1/r` while its circumference grows as `r` -- the angular
        cell size must grow at least like `r²`, hence `nb_sectors ∝ ( S/r )²`.

        Quantitative consequence: even on a domain ten times wider than the field of view, the halo
        stays at a few dozen DOFs. This is intended (see the module docstring).
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

    # -- projection operator -----------------------------------------------

    def _cell_values( self, a: float, b: float, phi0: float, phi1: float ) -> np.ndarray:
        """Projected density `[ nb_angles, nb_coarse ]` of a cell (ring sector) of density 1.

        EXACT in the radial direction, quadrature in the angular direction. At fixed φ, the
        radial segment `r ∈ [a,b]` projects onto `s = r·c` with `c = cos( θ − φ )`; the mass
        element `r dr dφ` becomes `( s/c )( ds/c ) dφ`, i.e. the EXACT density `|s|/c²·dφ` on
        the interval bounded by `a·c` and `b·c` (total mass `dφ( b² − a² )/2`, the area of the
        slice -- this is the check of the computation). Integrating analytically in r avoids the
        `n_r` cost factor of a 2D quadrature, and makes exact the `√( a² − s² )` singularity of the
        inner edge, which falls right at the ends of the detector.

        What remains is to sum these `α|s|` ramps over the bins. Depositing them bin by bin would cost the
        number of covered bins; we therefore go through the PRIMITIVE, of which each ramp only modifies
        two edge indices. With `β = sign(c)·dφ/( 2c² )`, `lo = min( a·c, b·c )` and
        `hi = max( a·c, b·c )`, the primitive of ONE ramp is written without an indicator:

            F( x ) = β·( clip( x, lo, hi )² − lo² )

        (zero for `x ≤ lo`, constant `β( hi² − lo² )` = the mass for `x ≥ hi`). By separating the
        finished / active / not yet started ramps, the sum over all ramps equals
        `F( x ) = C( x ) + Q( x )·x² − L( x )` with `Q = Σ_{lo≤x≤hi} β`, `L = Σ_{lo≤x} β·lo²` and
        `C = Σ_{hi<x} β·hi²` -- three STEP functions, hence three `bincount`s on the edge
        indices followed by a `cumsum`. O(1) cost per node, independent of the number of bins.

        The jump of each is carried by the FIRST EDGE TO THE RIGHT of its position (`ceil`, not
        `round`: rounding would move one ramp out of two by half a bin, and the error, propagated
        by the `cumsum` then multiplied by `x²`, would contaminate the whole profile). It is this choice that
        makes the result EXACT and not approximate: `L` and `C` keeping the true `lo²`/`hi²` and not
        rounded values, we have `F( e ) = β( e² − lo² )` at the edge next to `lo`, hence the mass
        of each bin is the exact integral of the ramp over that bin.

        `β` blows up when `c → 0` (the segment projects onto a point). Nothing DIVERGES
        for all that -- `β·lo² = dφ·a²/2`, `β·hi² = dφ·b²/2` and `β·x²` stay finite, `x` being squeezed
        between `lo` and `hi` -- but `F = C + Q·x² − L` becomes a difference of huge terms,
        and the resulting catastrophic cancellation ruins the whole profile (measured: negative
        total mass for certain values of `phi_per_bin`, those that align a node on
        `θ ± π/2`). We therefore bound `|c|` from below, at the value that makes the ramp a thousand times
        narrower than a bin: beyond that its exact position has no meaning at this
        resolution, and `β` stays in a range where the subtraction is exact.

        The indices are clipped to `[ 0, nb_coarse+1 ]`: a ramp entirely to the LEFT of the
        detector ends at edge 0 (mass already spent, no bin touched), a ramp
        entirely to the RIGHT lands in the overflow index `nb_coarse+1`, never read.
        """
        # angular step: `phi_per_bin` ramp endpoints per coarse bin (the outermost node of the
        # cell, at `b`, sets the step). Below that, the sum of the ramps ripples at
        # the bin scale, whereas the halo footprint is smooth.
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

        W = self.nb_coarse + 2                                             # +1 edge, +1 overflow
        base = np.arange( self.nb_angles )[ :, None ] * W
        ml = self.nb_angles * W

        def edge_of( pos ):
            """(Flattened) index of the first edge to the right of `pos`."""
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

        F = C + Q * self.coarse_edges ** 2 - L                             # primitive at the edges
        return ( F[ :, 1: ] - F[ :, :-1 ] ) / self.coarse_dw               # mass per bin -> density

    # -- estimation --------------------------------------------------------

    def _smoothness_rows( self ) -> np.ndarray:
        """Differences between ANGULARLY neighboring sectors of the same ring (cyclic).

        No radial coupling: the rings do not have the same number of sectors, the pairing
        would be arbitrary -- and positivity + the coarseness of the mesh already regularize
        a lot in that direction.
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
        """Fits the cell densities to the `residual` (measured MINUS interior model),
        `[ nb_angles, nb_bins ]` as a DENSITY. Updates `weights` and returns `self`.

        The residual is grouped onto the coarse grid before fitting -- that is where we gain
        the most: the residual noise (undersampling of the cloud, cf. `interior_values`) is
        white across bins, and a regression with a few dozen DOFs on `nb_angles x nb_coarse`
        equations divides it by a factor `√( nb_angles·nb_coarse / nb_cells )`, typically 70.

        `target_mass` (`[ nb_angles ]`, optional): the mass the halo must make VISIBLE at
        each angle, i.e. `∫p_θ − M_in`. This is the most solid anchor of the problem -- it comes
        directly from the data, without going through the shape of the profiles -- and it conditions the
        fit much better. `mass_weight` sets its relative weight.

        `ridge` / `smooth`: Tikhonov, and intra-ring angular smoothing (`_smoothness_rows`).
        Each block is normalized by its own Frobenius norm, so the three weights are
        dimensionless and comparable with each other.

        The solve is a POSITIVE least squares (`scipy.optimize.nnls`): positivity is not
        cosmetic, it is what prevents the halo from digging into interior signal.
        """
        res = np.asarray( residual, dtype = float )
        if res.shape != ( self.nb_angles, self.nb_bins ):
            raise ValueError( f"residual must have shape [ { self.nb_angles }, { self.nb_bins } ], "
                              f"got { res.shape }" )
        coarse = res.reshape( self.nb_angles, self.nb_coarse, self.group ).mean( axis = 2 )

        blocks = [ ( self.operator.reshape( self.nb_cells, -1 ).T, coarse.ravel(), 1.0 ) ]

        if target_mass is not None:
            # visible mass per angle of a cell of density 1: [ nb_angles, nb_cells ]
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

    # -- outputs -----------------------------------------------------------

    def values( self ) -> np.ndarray:
        """Footprint of the halo on the sinogram, `[ nb_angles, nb_bins ]`, as a density.

        The return to the fine grid is a simple repetition (`np.repeat`): constant per packet,
        hence of exact mass. The staircase it introduces is second order -- the footprint
        varies little from one coarse bin to the next, which is the whole premise of the module.
        """
        coarse = np.tensordot( self.weights, self.operator, axes = ( 0, 0 ) )
        return np.repeat( coarse, self.group, axis = 1 )

    def visible_mass( self ) -> np.ndarray:
        """Mass of the halo falling INSIDE the detector, per angle, `[ nb_angles ]`."""
        return self.values().sum( axis = 1 ) * self.dw

    def mass( self ) -> float:
        """Total mass of the halo (part of which is visible at no angle)."""
        return float( self.weights @ self.areas )

    def corrected( self, sinogram: Sinogram | None = None ) -> Sinogram:
        """The sinogram stripped of the halo footprint, clipped at 0.

        The clipping slightly breaks the mass accounting; this is harmless here (the
        values concerned are noise around zero) and it guarantees a valid target density
        for `SdotPlan1d`.
        """
        sino = sinogram if sinogram is not None else self.sinogram
        out = Sinogram( nb_angles = self.nb_angles, nb_bins = self.nb_bins,
                        extent = sino.extent, detector_center = sino.detector_center )
        out.values = np.clip( np.asarray( sino.values ) - self.values(), 0.0, None )
        return out


# -- projection of the interior cloud --------------------------------------


def interior_values( sinogram: Sinogram, points, mass: float, radius: float | None = None,
                     max_points: int | None = None, seed: int = 0 ) -> np.ndarray:
    """Density `[ nb_angles, nb_bins ]` projected by the cloud `points`, carrying the total mass
    `mass` at each angle.

    `mass` is supplied from OUTSIDE because the cloud has none: `SdotPlan1d` normalizes its two
    distributions, so the reconstruction only fixes the SHAPE. It is `alternate` that decides the
    interior mass (see its docstring).

    `radius`: `None` for diracs (deposited linearly on the two neighboring bins, mass
    conserved), otherwise disks of that radius (`DiskProjector`, the same projection as the one
    minimized by `DiskModel`).

    `max_points`: the cloud is SUBSAMPLED beyond that. This residual is only used to give a shape to
    the halo, at its coarse resolution and via a regression with a few dozen DOFs: putting in
    the 1e7 diracs of a fine reconstruction would be costly for nothing (cf. `Halo.fit`).
    """
    pts = np.asarray( points, dtype = float )
    if pts.ndim != 2 or pts.shape[ 1 ] != 2:
        raise ValueError( f"points must have shape [ n, 2 ], got { pts.shape }" )
    n = len( pts )
    if n == 0:
        return np.zeros( ( int( sinogram.nb_angles.value ), sinogram.nb_bins_host ) )
    if max_points is not None and n > max_points:
        pts = pts[ np.random.default_rng( seed ).choice( n, max_points, replace = False ) ]

    if radius is not None:
        # density of a disk of density 1 -> we renormalize so that each disk carries mass/len
        vals = np.asarray( DiskProjector( sinogram, radius = radius ).values( pts ) )
        return vals * ( mass / ( len( pts ) * np.pi * radius * radius ) )

    nb_angles, nb_bins = len( sinogram.angles ), sinogram.nb_bins_host
    out = np.zeros( nb_angles * nb_bins )
    per_point = mass / ( len( pts ) * sinogram.dw )                        # density, not mass

    # angle slices: the array of projected positions is [ nb_angles, nb_points ], quickly
    # larger than everything else (600 angles x 5e5 points = 2.4 GB).
    chunk = max( 1, int( 2e7 // max( 1, len( pts ) ) ) )
    for k0 in range( 0, nb_angles, chunk ):
        normals = sinogram.normals[ k0 : k0 + chunk ]                      # [ na, 2 ]
        s = normals @ pts.T                                                # [ na, nb_points ]
        x = ( s - sinogram.s_min ) / sinogram.dw - 0.5                     # in CENTER indices
        i0 = np.floor( x ).astype( int )
        w1 = x - i0
        base = ( k0 + np.arange( len( normals ) ) )[ :, None ] * nb_bins
        for idx, w in ( ( i0, 1.0 - w1 ), ( i0 + 1, w1 ) ):
            ok = ( idx >= 0 ) & ( idx < nb_bins )
            flat = ( base + np.clip( idx, 0, nb_bins - 1 ) )[ ok ]
            out += np.bincount( flat, ( w * per_point )[ ok ], nb_angles * nb_bins )
    return out.reshape( nb_angles, nb_bins )


# -- the alternation -------------------------------------------------------


def alternate( sinogram: Sinogram, solve, *, outer_radius: float | None = None,
               halo: Halo | None = None, nb_outer: int = 3, nb_points: int | None = None,
               interior_mass: float | None = None, radius: float | None = None,
               max_residual_points: int | None = 500_000, verbose: bool = False,
               halo_kwargs: dict | None = None,
               **recon_kwargs ) -> tuple[ Reconstruction, Halo ]:
    """Alternates HALO estimation and INTERIOR reconstruction, and returns both.

    "Remove from the sinogram what was found outside" is circular taken literally:
    to know the exterior one must know the interior. We therefore alternate, starting from a ZERO
    halo -- the first reconstruction is the bad one (the voids are filled), but
    the interior, confined to the field of view and of limited capacity, cannot reproduce an
    angularly inconsistent sinogram: what it leaves in the residual IS the unexplainable part of the
    data, exactly the signal the halo is looking for. Two or three passes are enough, the two
    models living at different scales.

        rec, halo = alternate( sino, lambda r: r.multiscale( 5000 ), outer_radius = 4.0 )

    `solve( rec )`: ONE complete interior solve, starting from the current cloud of `rec` (which is
    therefore warm-started from one pass to the next) and on the sinogram corrected at that moment. Must return
    `rec` -- typically `lambda r: r.multiscale( n )` or `lambda r: r.diracs().disks( radius )`.

    `interior_mass`: the mass to attribute to the INTERIOR. By default `min_θ ∫p_θ`, which is the
    exact bound (`∫p_θ = M_in + M_out( θ )` with `M_out ≥ 0`) -- and the only choice that guarantees
    a residual of positive mass at all angles, hence something to fit for a
    positively-constrained halo. It is an UPPER bound, attained only if there is an angle where
    the object fits entirely in the detector; otherwise the halo is underestimated, and this is the
    parameter to lower. `mass_profile` gives the `∫p_θ` curve to judge it, and `void_fraction`
    the criterion: the right split is the one that maximizes the void.

    `radius`: disk radius for the projection of the cloud in the residual (`None` = diracs). To be
    matched to the model that `solve` plays.

    `nb_points`: initial draw, for a `solve` that does not take care of it itself (`multiscale` does,
    `diracs`/`disks` do not).

    `recon_kwargs` -> `Reconstruction`. `extent` there defaults to that of the DETECTOR and not that
    of the object: the interior must stay in the field of view, the halo carries the rest.
    """
    if halo is None:
        if outer_radius is None:
            raise ValueError( "provide `halo`, or `outer_radius` to build one" )
        halo = Halo( sinogram, outer_radius = outer_radius, **( halo_kwargs or {} ) )

    raw = np.asarray( sinogram.values, dtype = float )
    per_angle = raw.sum( axis = 1 ) * sinogram.dw
    m_in = float( per_angle.min() ) if interior_mass is None else float( interior_mass )
    target = np.maximum( per_angle - m_in, 0.0 )                           # VISIBLE mass of the halo

    if verbose:
        print( f"[halo] { halo }" )
        print( f"[halo] mass per angle: min={ per_angle.min():.4g} max={ per_angle.max():.4g} "
               f"-> M_in={ m_in:.4g}, visible halo <= { target.max():.4g}" )

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
            print( f"[halo] pass { it }: halo mass { halo.mass():.4g} "
                   f"(visible { got.min():.4g}..{ got.max():.4g}, target { target.min():.4g}..{ target.max():.4g}), "
                   f"{ int( ( halo.weights > 0 ).sum() ) }/{ halo.nb_cells } active cells" )

    return rec, halo


def scan_interior_mass( halo: Halo, points, masses = None, radius: float | None = None,
                        max_points: int | None = 500_000, sinogram: Sinogram | None = None,
                        **fit_kwargs ) -> dict:
    """Sweeps the interior mass `M_in` and returns, for each, what the halo makes of it.

    `M_in` is the least determined parameter of the problem (see `alternate`): the default bound
    `min_θ ∫p_θ` is only attained if there is an angle where the object fits entirely in the
    detector, which is false as soon as the part overflows in ALL directions. Measured on
    `experiments/halo_demo`, it then overestimates `M_in` by 50%, and the halo recovers 1.5 of mass
    instead of 6.8 -- whereas at the right value it recovers it to within 0.8%. This is by far the
    first source of error of the module, ahead of the fineness of the mesh.

    The sweep is CHEAP: the projection of the interior is linear in `M_in`, so it is computed
    only once, and each point of the sweep is just an NNLS with a few dozen
    unknowns. Returns a dict of arrays (`masses`, `halo_mass`, `dispersion`, `nb_active`)
    -- `dispersion` being the relative standard deviation of `∫q_θ` after correction, whose minimum brackets
    the right value (shallow dip: to be used as a hint, not as an estimator).

    To be cross-checked with `void_fraction` on the resulting reconstruction, which is the true criterion: the right
    split is the one that makes the voids emptiest.
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
    halo.weights = keep                                                    # the sweep fits nothing
    return dict( masses = masses, **{ k: np.array( v ) for k, v in out.items() } )


# -- diagnostics -----------------------------------------------------------


def mass_profile( sinogram: Sinogram ) -> np.ndarray:
    """`∫p_θ` per angle, `[ nb_angles ]` -- the DIRECT measure of the leakage.

    Constant to within the noise precision = the object fits in the detector, nothing to correct. Its
    variation is exactly `M_out( θ )`, the exterior mass seen at angle θ.
    """
    return np.asarray( sinogram.mass(), dtype = float )


def void_fraction( points, extent: float, nb_cells: int | None = None, center = ( 0.0, 0.0 ) ) -> float:
    """Fraction of the cells of an `nb_cells²` grid (covering `extent` around `center`) that the
    cloud leaves EMPTY -- the criterion that motivates this whole module.

    COMPARATIVE diagnostic: for a cloud of the same size and the same grid, the higher it is, the better the
    voids have been preserved. It also counts the background outside the object, and only makes sense at
    EQUAL `nb_cells` -- hence the `√n` default: too fine, the grid saturates (`n` points can only occupy
    `n` cells out of `nb_cells²`, any cloud looks equally empty); too coarse, everything is
    occupied. With `√n` cells per side, a well-spread cloud fills about 63%.
    """
    pts = np.asarray( points, dtype = float )
    nb_cells = max( 2, int( np.sqrt( len( pts ) ) ) ) if nb_cells is None else int( nb_cells )
    lo = np.asarray( center, dtype = float ) - extent / 2
    idx = np.floor( ( pts - lo ) / extent * nb_cells ).astype( int )
    ok = np.all( ( idx >= 0 ) & ( idx < nb_cells ), axis = 1 )
    flat = idx[ ok, 0 ] * nb_cells + idx[ ok, 1 ]
    return float( 1.0 - np.count_nonzero( np.bincount( flat, minlength = nb_cells ** 2 ) ) / nb_cells ** 2 )
