"""Synthetic (binary) lung slice: two solid lobes pierced by ~1e4 alveoli.

Everything is built ANALYTICALLY in the Radon domain, by superposition of disks
(`Sinogram.add_disk`): a lobe (density +1) then each alveolus (density -1, an air hole) --
the Radon transform is linear, so the resulting sinogram is EXACT, whatever
`nb_angles`/`nb_bins` -- never any pixel discretization of the original image.

The alveoli are placed on a jittered hexagonal grid: the jitter is bounded
analytically (see `_hex_sites`) to guarantee that no pair can overlap, whatever
the random realization -- the phantom stays rigorously binary (0/1 after taking
the sign: lobe minus alveoli), not just "in general".
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from .. import dirac_fused
from ..Reconstruction import Reconstruction
from ..Sinogram import Sinogram
from ..optimizers import LBFGS, GradientDescentLineSearch, FusedLBFGS, SubspaceNewtonLBFGS


def _hex_sites( lobes, max_radius, spacing_factor = 2.15, jitter_factor = 0.02, seed = 0 ):
    """Candidate centers for disks of radius <= `max_radius`, on a jittered hexagonal grid,
    filtered to lie inside `lobes` (list of `(center, radius)`).

    The per-point jitter is bounded by `jitter_factor * spacing` (uniform per axis, amplitude
    `a`) so that even in the worst case (two neighbors jittered toward each other), the
    residual distance stays >= 2*max_radius: `spacing - 2*a*sqrt(2) >= 2*max_radius`. With
    `spacing = spacing_factor*max_radius` and `jitter_factor <= (spacing_factor - 2) / (2*sqrt(2)
    * spacing_factor)`, this guarantee is deterministic (independent of the draw) -- no a posteriori
    rejection pass needed.
    """
    rng = np.random.default_rng( seed )
    spacing = spacing_factor * max_radius
    row_h = spacing * np.sqrt( 3 ) / 2
    jitter_amp = jitter_factor * spacing

    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes ) + spacing

    rows = int( 2 * bound / row_h ) + 2
    cols = int( 2 * bound / spacing ) + 2

    ii, jj = np.meshgrid( np.arange( -rows, rows ), np.arange( -cols, cols ), indexing = "ij" )
    y = ii.ravel() * row_h
    x = jj.ravel() * spacing + np.where( ii.ravel() % 2 != 0, spacing / 2, 0.0 )
    pts = np.stack( [ x, y ], axis = 1 )
    pts = pts[ ( np.abs( pts[ :, 0 ] ) <= bound ) & ( np.abs( pts[ :, 1 ] ) <= bound ) ]

    pts = pts + ( rng.random( pts.shape ) * 2 - 1 ) * jitter_amp

    inside = np.zeros( len( pts ), dtype = bool )
    for center, radius in lobes:
        d = np.linalg.norm( pts - center[ None, : ], axis = 1 )
        inside |= d <= ( radius - max_radius * 1.05 )
    return pts[ inside ], rng


def make_lung_phantom(
    nb_angles: int = 600,
    nb_bins: int = 2000,
    extent: float = 44.0,
    nb_alveoli: int | None = None,
    alveolus_radius: float | None = None,
    scale: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
):
    """Builds the phantom: two lobes (density-1 disks) pierced by alveoli (density -1 disks,
    non-overlapping -- see `_hex_sites`). Returns `( sinogram, lobes, alveoli )` --
    `lobes`/`alveoli` = lists of `( center, radius )`, the analytical "ground truth" (a pixel
    image never exists: only the sinogram and these disks exist).

    At `scale = 1.0`, the two lobes fit inside the detector ([-extent/2, extent/2], margin ~2
    units) WITHOUT overlapping each other (otherwise their superposition would exceed density 1,
    breaking the binary character) -- values calibrated for extent=44, to be readjusted if `extent`
    changes strongly. `scale > 1` enlarges lobes AND alveoli (`alveolus_radius` follows by default,
    unless explicitly provided) WITHOUT changing `extent`: the object then EXCEEDS the detector --
    see `Sinogram.debias_and_equalize_mass` for the associated correction (mass per angle not
    constant when part of the shadow leaves the visible window).

    `alveolus_radius=None` (default) equals `0.075 * scale` -- the alveoli grow with the
    phantom, to occupy the same relative fraction of the area.

    `nb_alveoli=None` (default) FILLS all available sites (maximal hexagonal packing
    compatible with `_hex_sites`, ~63% of the lobe area as air at the default
    density) -- real alveoli occupy a large fraction of the lung volume,
    not just a few scattered holes; pass an integer for an explicit count (smaller,
    more readable to the eye -- useful for a pedagogical example).
    """
    if alveolus_radius is None:
        alveolus_radius = 0.075 * scale

    lobes = [
        ( np.array( [ 0.0, 0.0 ] ) * scale, 0.5 * extent * scale ),
    ]

    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    for center, radius in lobes:
        sino.add_disk( center = center, radius = radius, density = 1.0 )

    sites, rng = _hex_sites( lobes, alveolus_radius, seed = seed )
    if nb_alveoli is None:
        nb_alveoli = len( sites )
    if len( sites ) < nb_alveoli:
        nb_alveoli = len( sites )
        print(
            f"only { len( sites ) } sites available for { nb_alveoli } alveoli -- "
            "reduce alveolus_radius or nb_alveoli, or enlarge the lobes" )
    rng.shuffle( sites )
    sites = sites[ :nb_alveoli ]
    # 0.8-1.0 (rather than 0.55-1.0): alveoli more uniform in size, denser filling --
    # real alveoli occupy a large fraction of the lung volume, not just scattered holes
    # in a block of tissue.
    radii = alveolus_radius * ( 0.8 + 0.2 * rng.random( nb_alveoli ) )

    t0 = time.time()
    for i, ( center, radius ) in enumerate( zip( sites, radii ) ):
        # density -1: digs an air hole in the lobe (density +1) -- the Radon transform is
        # linear so the superposition stays exact, whatever the order of addition.
        sino.add_disk( center = center, radius = float( radius ), density = -1.0 )
        if verbose and ( i + 1 ) % 2000 == 0:
            print( f"  alveolus { i + 1 } / { nb_alveoli } ({ time.time() - t0:.1f}s)" )

    alveoli = list( zip( sites, radii ) )
    if verbose:
        print( f"phantom: { len( alveoli ) } alveoli, { time.time() - t0:.1f}s" )
    return sino, lobes, alveoli


def plot_phantom( lobes, alveoli, extent, ax = None, bound = None, show_detector = False ):
    """Draws the ground truth (solid lobes, alveoli in white) -- never rasterized, just
    the analytical disks themselves.

    `bound`: half-width of the displayed axes. Defaults to `extent / 2` (detector window) --
    but the PHYSICAL object (the lobes) may be larger than that (`scale > 1`, see
    `make_lung_phantom`): pass the bounding radius of the lobes explicitly so as not to
    clip them in the display. `show_detector` then draws the detector window as dotted lines,
    to visualize what the sensor actually sees.
    """
    if ax is None:
        _, ax = plt.subplots( figsize = ( 6, 6 ) )
    if bound is None:
        bound = extent / 2
    ax.add_collection( PatchCollection(
        [ Circle( c, r ) for c, r in lobes ], facecolor = "black", edgecolor = "none" ) )
    ax.add_collection( PatchCollection(
        [ Circle( c, r ) for c, r in alveoli ], facecolor = "white", edgecolor = "none" ) )
    if show_detector:
        ax.axvline( -extent / 2, color = "red", linestyle = "--", linewidth = 1 )
        ax.axvline(  extent / 2, color = "red", linestyle = "--", linewidth = 1 )
    ax.set_xlim( -bound, bound )
    ax.set_ylim( -bound, bound )
    ax.set_aspect( "equal" )
    ax.set_title( f"{ len( alveoli ) } alveoli (ground truth)" )
    return ax


def plot_sinogram( sino, ax = None ):
    if ax is None:
        _, ax = plt.subplots( figsize = ( 6, 6 ) )
    ax.imshow( np.asarray( sino.values ), aspect = "auto", cmap = "gray",
               extent = [ sino.s_min, sino.s_min + sino.extent, int( sino.nb_angles.value ), 0 ] )
    ax.set_xlabel( "detector s" )
    ax.set_ylabel( "angle k" )
    ax.set_title( "sinogram" )
    return ax


def run( nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000, max_iter = 60,
         out = "tmp/lung_alveoli.png", plot_max_points = 300_000 ):
    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, lobes, alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    fig, axes = plt.subplots( 1, 3, figsize = ( 18, 6 ) )
    plot_phantom( lobes, alveoli, extent = sino.extent, ax = axes[ 0 ] )
    plot_sinogram( sino, ax = axes[ 1 ] )

    print( f"reconstruction ({ nb_diracs } Diracs, LBFGS max_iter={ max_iter })..." )
    rec = Reconstruction( sino, max_iter = max_iter, ftol = 1e-10, verbose = True )
    rec.random_points( nb_diracs, seed = 1 )
    t0 = time.time()
    losses = []
    def callback( step, pos ):
        if step % 10 == 0 or step == -1:
            l = rec.loss( points = pos )
            losses.append( ( step, l ) )
            print( f"  step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
    # backend = "fused": fused kernel (`dirac_fused.diracs_cost_grad`), ~10-30x faster
    # than the default pure Jax path at this scale (see
    # `benchmarks/execution_speed/benchmark_fused.py`).
    rec.diracs( callback = callback, backend = "fused" )
    print( f"reconstruction finished in { time.time() - t0:.1f}s" )

    pos = rec.positions
    if len( pos ) > plot_max_points:
        # subsampling for display only -- the reconstruction itself used all
        # `nb_diracs` points, this only affects the matplotlib rendering
        # (a scatter of 1e7 points would be unmanageable in memory/render time).
        idx = np.random.default_rng( 0 ).choice( len( pos ), plot_max_points, replace = False )
        pos = pos[ idx ]
    axes[ 2 ].plot( pos[ :, 0 ], pos[ :, 1 ], '.', markersize = 0.5, color = "black" )
    axes[ 2 ].set_xlim( -sino.extent / 2, sino.extent / 2 )
    axes[ 2 ].set_ylim( -sino.extent / 2, sino.extent / 2 )
    axes[ 2 ].set_aspect( "equal" )
    axes[ 2 ].set_title( f"reconstruction ({ nb_diracs } Diracs)" )

    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )


def run_subspace( nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000, max_iter = 60,
                   max_dirs = 5, out = "tmp/lung_alveoli_subspace.png",
                   out_html = "tmp/lung_alveoli_subspace_{name}.html" ):
    """Compares `FusedLBFGS` (scipy L-BFGS-B, internal curvature memory) against
    `SubspaceNewtonLBFGS` (EXACT Newton in the subspace spanned by the last normalized
    gradients -- see its docstring in `optimizers.py` and `dirac_fused.subspace_hessian` for the
    derivation): SAME phantom, SAME starting cloud (`seed=1`), SAME iteration budget for
    both -- only the trajectory differs. Used to evaluate the proposal tested in this
    experiment (assisting/making the line search multi-dimensional), not to replace the
    default optimizer of `Reconstruction.diracs`.

    Outputs: `out` (PNG, superimposed loss curves, as before) AND, per optimizer, a page
    `out_html` (`{name}` replaced by the optimizer name -- see `export_positions_html`) that
    REPLAYS the trajectory (`record=True`) -- to compare BY EYE how each solver moves
    the cloud, not just the final loss value.
    """
    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, _lobes, _alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    optimizers = {
        "FusedLBFGS": lambda: FusedLBFGS( max_iter = max_iter, ftol = 1e-10 ),
        "SubspaceNewtonLBFGS": lambda: SubspaceNewtonLBFGS(
            sinogram = sino, max_dirs = max_dirs, max_iter = max_iter, ftol = 1e-10 ),
    }

    results = {}
    for name, make_optimizer in optimizers.items():
        print( f"reconstruction ({ nb_diracs } Diracs, { name }, max_iter={ max_iter })..." )
        rec = Reconstruction( sino, record = True, verbose = True )
        rec.random_points( nb_diracs, seed = 1 )
        t0 = time.time()
        losses = []
        def callback( step, pos, losses = losses, t0 = t0, name = name, rec = rec ):
            l = rec.loss( points = pos )
            losses.append( ( step, l, time.time() - t0 ) )
            if step % 10 == 0 or step == -1:
                print( f"  [{ name }] step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
        rec.diracs( callback = callback, backend = "fused", optimizer = make_optimizer() )
        print( f"[{ name }] finished in { time.time() - t0:.1f}s" )
        results[ name ] = losses

        html_path = out_html.format( name = name )
        rec.export_html( html_path, title = f"{ name } ({ nb_diracs } Diracs)" )
        print( f"page saved: { html_path }" )

    fig, ax = plt.subplots( figsize = ( 7, 5 ) )
    for name, losses in results.items():
        steps = [ s for s, l, t in losses ]
        vals  = [ l for s, l, t in losses ]
        ax.plot( steps, vals, label = name )
    ax.set_xlabel( "iteration" )
    ax.set_ylabel( "loss" )
    ax.set_yscale( "log" )
    ax.legend()
    ax.set_title( f"FusedLBFGS vs SubspaceNewtonLBFGS ({ nb_diracs } Diracs)" )
    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )
    return results


def _parabolic_bracket( model, p, cost0, g, a, max_tries = 30 ):
    """Looks for `a` such that the midpoint (`a/2`) is the lowest of the three costs at
    `(0, a/2, a)` along `-g` -- core of `_parabolic_line_search`, isolated here to be
    reusable by the diagnostic (`run_parabola_diagnostic`), which needs the bracket AND the
    number of tries (a measure of the quality of the initial guess), not just the final minimum.

    ONLY the cost is evaluated at the candidate points (`model.value`, fused kernel WITHOUT
    the gradient computation -- see `dirac_fused.diracs_cost` -- the gradient would be discarded
    there anyway).

    - `cost_half` the smallest of the three -> bracket found, we stop.
    - `cost_a` the smallest -> it is still decreasing at the edge `a`, the minimum is further: we
      double `a` (reusing the evaluation already done at `a` as the new midpoint).
    - `cost0` (the start) the smallest -> `a` is too large: we halve it (same
      reuse, lower side).
    Each expansion/reduction therefore costs only ONE cost evaluation, not two.

    Returns `(a, cost_half, cost_a, tries)` -- `tries` = number of expansions/reductions before the
    bracket (0 = the initial guess was already good).
    """
    cost_half = model.value( p - ( a / 2 ) * g )
    cost_a    = model.value( p - a * g )

    tries = 0
    for tries in range( max_tries ):
        if cost_half <= cost0 and cost_half <= cost_a:
            break
        if cost_a <= cost_half and cost_a <= cost0:
            a *= 2
            cost_half, cost_a = cost_a, model.value( p - a * g )
        else:
            a /= 2
            cost_a, cost_half = cost_half, model.value( p - ( a / 2 ) * g )

    return a, cost_half, cost_a, tries


def _parabola_vertex( cost0, cost_half, cost_a, a ):
    """Abscissa of the minimum of the parabola through the 3 equidistant points `(0, cost0)`,
    `(a/2, cost_half)`, `(a, cost_a)`, folded into `[0, a]` (protection against a `denom` close
    to 0 -- nearly flat parabola, the vertex position becomes numerically unstable)."""
    h = a / 2
    denom = cost0 - 2 * cost_half + cost_a
    if denom <= 0:
        return h
    return min( max( h + h * ( cost0 - cost_a ) / ( 2 * denom ), 0.0 ), a )


def _parabola_eval( cost0, cost_half, cost_a, a, x ):
    """(Vectorized) value of THIS SAME parabola at `x` -- Lagrange interpolation through the 3
    equidistant points `(0, cost0)`, `(h, cost_half)`, `(2h, cost_a)` with `h = a/2`. Used
    ONLY for the diagnostic plot (`run_parabola_diagnostic`) -- `_parabola_vertex`
    does not need it, a closed formula suffices for the vertex alone."""
    h = a / 2
    x = np.asarray( x )
    l0 = ( x - h ) * ( x - 2 * h ) / ( 2 * h * h )
    l1 = x * ( x - 2 * h ) / ( -h * h )
    l2 = x * ( x - h ) / ( 2 * h * h )
    return cost0 * l0 + cost_half * l1 + cost_a * l2


def _parabolic_line_search( model, p, cost0, g, a, max_tries = 30 ):
    """Step along `-g` chosen by fitting a parabola on 3 points (0, a/2, a) instead
    of an Armijo backtracking -- see `_parabolic_bracket` for the bracketing details and
    `_parabola_vertex` for the retained vertex.

    `a`: starting guess provided by the caller -- `run_two_phase_switch` passes `2 *` the
    coefficient ACCEPTED at the previous step (see `a_min` in the output here) rather than starting from a
    heuristic at each step: the optimal step generally varies only slowly from one iteration to the
    next, and starting from its double always leaves a chance of re-detecting an expansion.

    Returns `(p_new, cost_new, g_new, a_min)` -- the gradient at `p_new` is RE-evaluated (needed
    for the next gradient step, `_parabolic_bracket` did not need it); `a_min` is the
    ACCEPTED coefficient, to pass on (doubled) as the guess for the next step.
    """
    a, cost_half, cost_a, _tries = _parabolic_bracket( model, p, cost0, g, a, max_tries )
    a_min = _parabola_vertex( cost0, cost_half, cost_a, a )

    cost_new, g_new = model.value_and_grad( p - a_min * g )
    return p - a_min * g, float( cost_new ), np.asarray( g_new ), a_min


def run_two_phase_switch(
        nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000,
        phase1_guess_disp_frac = 0.01,
        phase1_disp_tol_frac = 1e-5, phase1_max_steps = 200,
        max_iter = 100, max_dirs = 5,
        out = "tmp/lung_alveoli_two_phase.png" ):
    """Sanity check for the hypothesis "coupling the search directions (`SubspaceNewtonLBFGS`)
    brings nothing as long as the OT assignment is not stabilized" (the subspace Hessian
    is only reliable with a frozen assignment, see the class docstring): BEFORE building a
    switching criterion based on assignment stability (more expensive to instrument), we switch
    here on a simpler criterion -- the max displacement per step, SAME convention as
    `disks_disp_tol_frac` elsewhere in this file, a fraction of `sino.extent`.

    Phase 1: gradient descent + step search by parabola fitting
    (`_parabolic_line_search`, NO Armijo backtracking -- see its docstring) written by hand
    here (NOT `GradientDescentLineSearch.minimize`, which has no early-stopping criterion -- see
    `optimizers.py`), + stop as soon as the max displacement of a point falls below
    `phase1_disp_tol_frac * sino.extent`. Uses `model.value_and_grad` (fused kernel)
    rather than Jax autodiff, to remain comparable to phase 2.
    Phase 2: standard `SubspaceNewtonLBFGS` starting from the resulting cloud, until the
    remaining `max_iter` budget is consumed.

    Compared with `SubspaceNewtonLBFGS` alone from the SAME start (same phantom, seed=1) over the SAME
    total budget `max_iter` of steps -- the hypothesis predicts a convergence at least as good (in
    steps AND in time, phase 1 being markedly cheaper per step than the subspace
    Hessian) for the two-phase version.

    Throwaway/exploratory: no HTML export (unlike `run_subspace`), just the loss curves
    vs iterations AND vs time -- sufficient for the question asked ("is it worth
    digging into a finer switching criterion, based on assignment stability?").
    """
    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, _lobes, _alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    def make_rec():
        rec = Reconstruction( sino, verbose = True )
        rec.random_points( nb_diracs, seed = 1 )
        return rec

    results = {}

    # -- baseline: SubspaceNewtonLBFGS alone from the start ---------------
    # name = "SubspaceNewtonLBFGS (alone)"
    # print( f"reconstruction ({ nb_diracs } Diracs, { name }, max_iter={ max_iter })..." )
    # rec = make_rec()
    # t0 = time.time()
    # losses = []
    # def callback( step, pos, losses = losses, t0 = t0, name = name, rec = rec ):
    #     l = rec.loss( points = pos )
    #     losses.append( ( step, l, time.time() - t0 ) )
    #     if step % 10 == 0 or step == -1:
    #         print( f"  [{ name }] step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
    # rec.diracs( callback = callback, backend = "fused", optimizer = SubspaceNewtonLBFGS(
    #     sinogram = sino, max_dirs = max_dirs, max_iter = max_iter, ftol = 1e-10 ) )
    # print( f"[{ name }] finished in { time.time() - t0:.1f}s" )
    # results[ name ] = losses

    # -- two phases: line search (bounded displacement) then SubspaceNewton --
    name = "line search then SubspaceNewtonLBFGS"
    print( f"reconstruction ({ nb_diracs } Diracs, { name })..." )
    rec = make_rec()
    model = rec.dirac_model()
    t0 = time.time()
    losses = []

    p = np.array( rec.points.raw )
    disp_tol = phase1_disp_tol_frac * sino.extent
    cost, g = model.value_and_grad( p )
    cost, g = float( cost ), np.asarray( g )
    losses.append( ( 0, cost, time.time() - t0 ) )
    print( f"  [phase1] step 0: loss = { cost:.6f} (start)" )

    # initial guess: MEAN displacement of a point = `phase1_guess_disp_frac * sino.extent`. At
    # the following steps, `_parabolic_line_search` returns the ACCEPTED coefficient -- we start from its
    # double rather than recomputing this heuristic at each step (the optimal step generally
    # varies only slowly from one iteration to the next).
    mean_norm = float( np.mean( np.linalg.norm( g, axis = -1 ) ) )
    a_guess = phase1_guess_disp_frac * sino.extent / mean_norm if mean_norm > 0 else 0.0

    phase1_steps = 0
    for step in range( 1, phase1_max_steps + 1 ):
        p_try, cost_try, g_try, a_used = _parabolic_line_search( model, p, cost, g, a_guess )
        a_guess = 2 * a_used

        disp = float( np.max( np.abs( p_try - p ) ) )
        p, cost, g = p_try, cost_try, g_try
        phase1_steps = step
        losses.append( ( step, cost, time.time() - t0 ) )
        if step % 10 == 0:
            print( f"  [phase1] step { step }: loss = { float( cost ):.6f}, "
                   f"max displacement = { disp:.4g} (threshold { disp_tol:.4g})" )
        if disp < disp_tol:
            print( f"  [phase1] stop: max displacement { disp:.4g} < threshold { disp_tol:.4g} "
                   f"after { step } steps" )
            break
    else:
        print( f"  [phase1] budget of { phase1_max_steps } steps exhausted without falling below the "
               "displacement threshold" )

    rec.set_points( model.wrap( p ) )
    print( f"[phase1] finished in { phase1_steps } steps ({ time.time() - t0:.1f}s), "
           f"loss = { float( cost ):.6f}" )

    # phase2_budget = max( max_iter - phase1_steps, 1 )
    # def callback2( step, pos, losses = losses, t0 = t0, phase1_steps = phase1_steps, rec = rec ):
    #     l = rec.loss( points = pos )
    #     losses.append( ( phase1_steps + step + 1, l, time.time() - t0 ) )
    #     if step % 10 == 0 or step == -1:
    #         print( f"  [phase2] step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
    # rec.diracs( callback = callback2, backend = "fused", optimizer = SubspaceNewtonLBFGS(
    #     sinogram = sino, max_dirs = max_dirs, max_iter = phase2_budget, ftol = 1e-10 ) )
    # print( f"[{ name }] finished in { time.time() - t0:.1f}s "
    #        f"({ phase1_steps } phase1 steps + up to { phase2_budget } phase2 steps)" )
    results[ name ] = losses

    rec.export_html( "tmp/two_phase_switch.html", animate = False )

    fig, axes = plt.subplots( 1, 2, figsize = ( 13, 5 ) )
    for name, losses in results.items():
        steps = [ s for s, l, t in losses ]
        vals  = [ l for s, l, t in losses ]
        times = [ t for s, l, t in losses ]
        axes[ 0 ].plot( steps, vals, label = name )
        axes[ 1 ].plot( times, vals, label = name )
    axes[ 0 ].set_xlabel( "iteration (phase1 counted as equivalent steps)" )
    axes[ 1 ].set_xlabel( "time (s)" )
    for ax in axes:
        ax.set_ylabel( "loss" )
        ax.set_yscale( "log" )
        ax.legend()
    fig.suptitle( f"SubspaceNewtonLBFGS alone vs. line search + SubspaceNewtonLBFGS ({ nb_diracs } Diracs)" )
    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )
    return results


def run_parabola_diagnostic(
        nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000,
        phase1_guess_disp_frac = 0.01, nb_steps = 5, nb_samples = 20,
        out = "tmp/lung_alveoli_parabola_diagnostic.png" ):
    """Diagnostic for the parabola step search (`_parabolic_line_search`, see
    `run_two_phase_switch`): on the FIRST `nb_steps` steps of phase 1 (where the loss varies
    the most, the comparison with the second-order model is the most telling there -- same reasoning as
    `run_disks_alpha_profile`), superimposes at each step:
    - the parabola ACTUALLY used to choose the step (3 points `(0, a/2, a)`, see
      `_parabolic_bracket`/`_parabola_vertex`);
    - the REAL cost sampled at `nb_samples` values along the SAME direction `-g`, via
      `model.value` (cost only, fused kernel WITHOUT gradient -- `dirac_fused.diracs_cost`,
      see its docstring) -- no gradient is needed to judge the second order, hence no need for its
      computation cost.
    Answers two questions: is the second-order model relevant here (does the parabola match
    the real curve near the retained minimum)? And does the `2 * a_min` guess for the next step fall
    in a zone where this model remains reasonable (instead of re-bracketing from scratch at each step)?

    Each panel goes from `0` to `1.3 * max(a, 2*a_min)` (wide enough to also show the zone of the
    next guess) and annotates `a_min` (retained step) and `2*a_min` (next guess) with vertical
    lines, plus `tries` (number of bracket expansions/reductions -- 0 = guess already good) in
    the title.
    """
    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, _lobes, _alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    rec = Reconstruction( sino, verbose = True )
    rec.random_points( nb_diracs, seed = 1 )
    model = rec.dirac_model()

    p = np.array( rec.points.raw )
    cost, g = model.value_and_grad( p )
    cost, g = float( cost ), np.asarray( g )

    mean_norm = float( np.mean( np.linalg.norm( g, axis = -1 ) ) )
    a_guess = phase1_guess_disp_frac * sino.extent / mean_norm if mean_norm > 0 else 0.0

    fig, axes = plt.subplots( 1, nb_steps, figsize = ( 4.5 * nb_steps, 4 ), squeeze = False )
    axes = axes[ 0 ]

    for step in range( nb_steps ):
        a, cost_half, cost_a, tries = _parabolic_bracket( model, p, cost, g, a_guess )
        a_min = _parabola_vertex( cost, cost_half, cost_a, a )

        x_max = 1.3 * max( a, 2 * a_min, 1e-12 )
        xs = np.linspace( 0.0, x_max, nb_samples )
        true_costs = np.array( [ model.value( p - x * g ) for x in xs ] )
        fit_costs = _parabola_eval( cost, cost_half, cost_a, a, xs )

        ax = axes[ step ]
        ax.plot( xs, true_costs, "o-", label = "real cost", color = "tab:blue" )
        ax.plot( xs, fit_costs, "--", label = "parabola (0, a/2, a)", color = "tab:orange" )
        ax.scatter( [ 0, a / 2, a ], [ cost, cost_half, cost_a ], color = "black", zorder = 3,
                    label = "fit points" )
        ax.axvline( a_min, color = "tab:green", linestyle = ":", label = "a_min (retained)" )
        ax.axvline( 2 * a_min, color = "tab:red", linestyle = ":", label = "2*a_min (next guess)" )
        ax.set_title( f"step { step } (tries={ tries })\ncost0={ cost:.4g}" )
        ax.set_xlabel( "coefficient a" )
        if step == 0:
            ax.set_ylabel( "cost" )
            ax.legend( fontsize = 8 )

        # advance to the next step EXACTLY like `_parabolic_line_search` (gradient RE-evaluated at
        # the retained point, `model.value` alone only served for the sampling above)
        p = p - a_min * g
        cost, g = model.value_and_grad( p )
        cost, g = float( cost ), np.asarray( g )
        a_guess = 2 * a_min

        true_min = float( np.min( true_costs ) )
        print( f"  [step { step }] a_min={ a_min:.4g}, tries={ tries }, "
               f"real cost at retained point={ cost:.6f} vs sampled min={ true_min:.6f} "
               f"(relative gap { abs( cost - true_min ) / max( abs( true_min ), 1e-30 ):.2%})" )

    fig.suptitle( f"Parabola vs real cost -- first { nb_steps } phase 1 steps ({ nb_diracs } Diracs)" )
    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )


def run_subspace_alpha_profile(
        nb_alveoli = 1_000, alveolus_radius = 0.75, nb_diracs = 10_000, max_iter = 60,
        max_dirs = 5, capture_step = None, alpha_range = ( -2.0, 2.0 ), nb_alpha = 61,
        out = "tmp/lung_alveoli_subspace_alpha_profile.png" ):
    """Diagnostic to understand why `SubspaceNewtonLBFGS` converges poorly (see its docstring):
    for ONE given step (`capture_step`, by default the LAST step played), plots the true loss
    along `x + alpha * step_dir` (`step_dir` = the subspace Newton step SOLVED at this step,
    BEFORE any shortening by Armijo backtracking) against what the local quadratic model
    predicts (`H`, `b` from `dirac_fused.subspace_hessian`) -- if the two curves diverge
    quickly after `alpha=0`, the "frozen assignment" model (see the docstring of `subspace_hessian`)
    is no longer a good approximation as soon as we move away from the current point, which explains a
    Newton step (`alpha=1`) that is systematically too long / rejected by Armijo.

    Capture via `SubspaceNewtonLBFGS.diag_callback`, called at EACH step -- `capture_step = None`
    therefore keeps the LAST one (overwritten at each iteration), an integer targets a specific step.
    """
    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, _lobes, _alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    rec = Reconstruction( sino, verbose = True )
    rec.random_points( nb_diracs, seed = 1 )

    diag = {}
    def diag_callback( step, x, cost, g, directions, m, H, b, a, step_dir, directional_deriv ):
        if capture_step is not None and step != capture_step:
            return
        diag.clear()
        diag.update(
            step = step, x = np.array( x ), cost = float( cost ), step_dir = np.array( step_dir ),
            linear = float( directional_deriv ), quad = float( a @ H @ a ), m = m,
        )

    optimizer = SubspaceNewtonLBFGS(
        sinogram = sino, max_dirs = max_dirs, max_iter = max_iter, ftol = 1e-10,
        diag_callback = diag_callback )
    print( f"reconstruction ({ nb_diracs } Diracs, SubspaceNewtonLBFGS, max_iter={ max_iter })..." )
    rec.diracs( backend = "fused", optimizer = optimizer )

    if not diag:
        raise RuntimeError( f"no step captured (capture_step={ capture_step } out of range?)" )

    x, step_dir, cost0 = diag[ "x" ], diag[ "step_dir" ], diag[ "cost" ]
    linear, quad = diag[ "linear" ], diag[ "quad" ]
    print( f"captured step: { diag[ 'step' ] } (subspace of dimension { diag[ 'm' ] }), "
           f"loss={ cost0:.6f}, directional derivative={ linear:.4g}, curvature={ quad:.4g}" )

    alphas = np.linspace( *alpha_range, nb_alpha )
    real_losses = np.array(
        [ dirac_fused.diracs_cost_grad( x + alpha * step_dir, sino )[ 0 ] for alpha in alphas ] )
    model_losses = cost0 + alphas * linear + 0.5 * alphas ** 2 * quad

    fig, ax = plt.subplots( figsize = ( 7, 5 ) )
    ax.plot( alphas, real_losses, "o-", markersize = 3, label = "real loss" )
    ax.plot( alphas, model_losses, "--", label = "quadratic model (Newton)" )
    ax.axvline( 0.0, color = "gray", linewidth = 0.8 )
    ax.axvline( 1.0, color = "gray", linewidth = 0.8, linestyle = ":", label = "alpha=1 (Newton step)" )
    ax.set_xlabel( "alpha (along the last Newton direction)" )
    ax.set_ylabel( "loss" )
    ax.legend()
    ax.set_title( f"step { diag[ 'step' ] }: real loss vs quadratic model" )
    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )
    return dict( alphas = alphas, real_losses = real_losses, model_losses = model_losses, step = diag[ "step" ] )


def run_disks_alpha_profile(
        nb_alveoli = 1_000, alveolus_radius = 0.75, nb_disks = 1_000, disk_radius = 0.5,
        nb_steps = 6, alpha_range = ( -2.0, 2.0 ), nb_alpha = 41, fd_h = 1e-3,
        out = "tmp/lung_alveoli_disks_alpha_profile.png" ):
    """Same diagnostic as `run_subspace_alpha_profile`, for peace of mind, but for the
    DISK model (`models.DiskModel`) instead of DIRACS -- does the same kind of gap
    between real loss and local quadratic model occur here too?

    Unlike the Dirac case (last step only -- the trajectory being already nearly
    stationary at that stage), here we profile the FIRST `nb_steps` steps (one curve per step): it is
    far from any minimum, where a local model is most likely to deviate from the real
    loss, that the comparison is the most telling.

    There is NO disk equivalent of `dirac_fused.subspace_hessian` (no fused kernel
    for `DiskModel`, see its docstring -- no `value_and_grad`), so `rec.disks()`
    runs with the standard scipy `LBFGS` (Jax autodiff), whose internal search direction
    (BFGS curvature memory) is NOT exposed by the Python API -- impossible to intercept "the
    step the model proposed before backtracking" as for `SubspaceNewtonLBFGS`. For each
    step, the "model" compared is therefore a GENERIC local Taylor expansion (NO
    analytical closed formula here -- Hessian evaluated NUMERICALLY) at the starting point of the step ACTUALLY
    accepted by LBFGS: linear term = true Jax gradient, quadratic term = directional second
    derivative by finite differences on the gradient (step `fd_h`, in alpha units).

    `nb_disks=1_000` (instead of the default `10_000` Diracs): the disk model is MUCH
    more expensive per evaluation (image sweep per angle slice, no fused kernel, see
    `models.DiskModel`) -- limiting ourselves to the first `nb_steps` steps (instead of a full
    convergence) keeps this bearable. `min_iter = max_iter = nb_steps` forces EXACTLY this
    number of steps (see `Reconstruction.disks` -- without it, LBFGS-B may conclude convergence
    after 0-1 steps on this model, flat directions).
    """
    import loom

    print( f"generating phantom ({ nb_alveoli } alveoli)..." )
    sino, _lobes, _alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, alveolus_radius = alveolus_radius )

    rec = Reconstruction( sino, radius = disk_radius, max_iter = nb_steps, ftol = 1e-10, verbose = True )
    rec.random_points( nb_disks, seed = 1 )
    model = rec.disk_model()

    t0 = time.time()
    history = []
    def callback( step, pos ):
        history.append( np.array( pos.raw ) )
        print( f"  step { step }: loss = { rec.loss( points = pos ):.6f} ({ time.time() - t0:.1f}s)" )

    print( f"reconstruction ({ nb_disks } disks, LBFGS, { nb_steps } forced steps)..." )
    rec.disks( radius = disk_radius, callback = callback, min_iter = nb_steps )
    nb_captured = len( history ) - 1
    print( f"{ nb_captured } steps obtained in { time.time() - t0:.1f}s" )

    def scalar_loss( q ):
        return model.cost( model.wrap( q ) ).value
    loss_j = loom.jit( scalar_loss )
    grad_j = loom.jit( loom.grad( scalar_loss ) )

    alphas = np.linspace( *alpha_range, nb_alpha )
    profiles = []
    for step in range( nb_captured ):
        x0, x1 = history[ step ], history[ step + 1 ]
        step_dir = x1 - x0

        cost0 = float( loss_j( x0 ) )
        g0 = np.asarray( grad_j( x0 ) )
        linear = float( np.sum( g0 * step_dir ) )
        # curvature = directional second derivative by CENTERED finite differences on the
        # gradient, directly in alpha units (no renormalization by the norm of step_dir
        # needed): f'(alpha) = grad(x0 + alpha*step_dir) . step_dir, so
        # f''(0) ~ (f'(h) - f'(-h)) / (2h) -- NUMERICAL, no analytical Hessian for this model.
        g_plus  = np.asarray( grad_j( x0 + fd_h * step_dir ) )
        g_minus = np.asarray( grad_j( x0 - fd_h * step_dir ) )
        quad = float( np.sum( ( g_plus - g_minus ) * step_dir ) ) / ( 2 * fd_h )

        real_losses = np.array( [ float( loss_j( x0 + alpha * step_dir ) ) for alpha in alphas ] )
        model_losses = cost0 + alphas * linear + 0.5 * alphas ** 2 * quad
        print( f"  step { step }: loss={ cost0:.6f}, directional derivative={ linear:.4g}, "
               f"curvature (finite diff.)={ quad:.4g}" )
        profiles.append( dict( step = step, cost0 = cost0, linear = linear, quad = quad,
                              real_losses = real_losses, model_losses = model_losses ) )

    ncols = 3
    nrows = -( -len( profiles ) // ncols )
    fig, axes = plt.subplots( nrows, ncols, figsize = ( 5 * ncols, 4 * nrows ), squeeze = False )
    for ax, prof in zip( axes.ravel(), profiles ):
        ax.plot( alphas, prof[ "real_losses" ], "o-", markersize = 2, label = "real loss" )
        ax.plot( alphas, prof[ "model_losses" ], "--", label = "quadratic model" )
        ax.axvline( 0.0, color = "gray", linewidth = 0.8 )
        ax.axvline( 1.0, color = "gray", linewidth = 0.8, linestyle = ":", label = "alpha=1 (step taken)" )
        ax.set_title( f"step { prof[ 'step' ] } (loss={ prof[ 'cost0' ]:.4g})" )
        ax.set_xlabel( "alpha" )
        ax.set_ylabel( "loss" )
        if prof is profiles[ 0 ]:
            ax.legend( fontsize = 8 )
    for ax in axes.ravel()[ len( profiles ): ]:
        ax.axis( "off" )
    fig.suptitle( f"disks ({ nb_disks }) -- real loss vs quadratic model, first { len( profiles ) } steps" )
    fig.tight_layout()
    fig.savefig( out, dpi = 150 )
    print( f"figure saved: { out }" )
    return profiles


def run_truncated( nb_alveoli = 1000, scale = 1.0, nb_diracs_final = 4992*4,
                    out_phantom = "tmp/lung_truncated_phantom.png",
                    out_reconstruction = "tmp/lung_truncated_reconstruction.html",
                    point_radius = 0.05, animate = True, polish_steps = 0, polish_lr = 20.0,
                    disks_min_iter = 10, disks_disp_tol_frac = 1e-3, disks_split_before = 1 ):
    """Object larger than the detector (`scale=2`, `extent` unchanged): the shadow of
    the object exceeds the visible window at certain angles, so the mass measured per angle is no
    longer constant. `Sinogram.debias_and_equalize_mass` corrects this (additive shift per angle,
    masses equalized before any rescale -- see its docstring); here we only show the ground
    truth and the reconstruction obtained from the CORRECTED sinogram (see git log for the
    comparison with the uncorrected raw sinogram, which produced an hourglass artifact).

    Output: `out_phantom` (PNG, ground truth -- few disks, matplotlib is perfectly fine)
    and `out_reconstruction` (standalone HTML, point cloud -- see `export_positions_html`,
    much more readable than a matplotlib scatter at `nb_diracs_final` points).

    `animate` (default True): captures a frame at EACH optimizer step of EACH stage of
    `Reconstruction.multiscale` (thanks to `record`) -- the HTML page then gets
    a time bar + a play/pause button to replay the movement/refinement of the
    points until convergence. `False` reverts to the historical behavior (a single frame, the
    final reconstruction).

    `polish_steps`: after `multiscale`, an additional gradient descent step
    WITH line search (`GradientDescentLineSearch`) on `nb_diracs_final` points --
    LBFGS sometimes gets stuck on this problem (not really differentiable, saddle points: scipy's
    line search stops without having really converged). The simple backtracking of
    `GradientDescentLineSearch` does not have these curvature requirements and can keep
    squeezing out loss where LBFGS stopped. `0` disables this step.

    `polish_lr`: the gradient at the point where LBFGS stopped is already small (residue of a
    quasi-critical point) -- `lr=1.0` only produces a displacement of the order of 1e-4 times
    the extent there, invisible in the display. Since backtracking is risk-free (it only
    REDUCES the step if `lr` overshoots, never the opposite), we deliberately start large; measured
    empirically, the Armijo condition only needs to back off from an `lr` of the order of
    100 on this problem -- 20 leaves margin while giving a step much greater than 1.0.

    `disks_min_iter`/`disks_disp_tol_frac`: each `disks()` stage (see `Reconstruction.disks`)
    inherits a cloud coming from `diracs()` -- essentially one Dirac per sinogram bin
    (`models.sinogram_diracs`). From this starting point, the disk loss often has
    FLAT directions (a disk can slide without changing the residual as long as it overlaps no one): without a
    safeguard, L-BFGS-B may conclude convergence there after 0-1 steps, while the CENTERS remain
    far from a regular arrangement. `disks_min_iter` forces this minimum number of steps;
    `disks_disp_tol_frac` (fraction of the stage's current radius) then stops as soon as the points
    stop moving significantly -- `None` disables this second criterion (only scipy's native `ftol`
    then stops the phase after `disks_min_iter`).

    `disks_split_before`: if > 1, SUBDIVIDES the cloud (see `Reconstruction.split`) right before
    each disk stage -- more centers for the same given radius give more degrees of freedom
    for local rearrangement, useful if the cloud inherited from the Diracs is too sparse for the
    requested radius and ends up LOCKED (overlaps imposed by the neighborhood). `1` (default) changes
    nothing -- to be enabled/tuned by hand if `disks_min_iter` alone is not enough.
    """
    print( f"generating oversized phantom (scale={ scale })..." )
    sino, lobes, alveoli = make_lung_phantom( nb_alveoli = nb_alveoli, scale = scale, alveolus_radius = 0.7, nb_bins = 2000 )
    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes ) * 1.05
    recon_extent = 2 * bound

    raw_mass = np.asarray( sino.mass() )
    corrected = sino # .debias_and_equalize_mass()
    corr_mass = np.asarray( corrected.mass() )
    print( f"raw mass per angle: min={ raw_mass.min():.1f} max={ raw_mass.max():.1f} "
           f"(std { raw_mass.std():.1f})" )
    print( f"corrected mass per angle: min={ corr_mass.min():.1f} max={ corr_mass.max():.1f} "
           f"(std { corr_mass.std():.2e})" )

    ax = plot_phantom( lobes, alveoli, extent = sino.extent, bound = bound, show_detector = True )
    ax.set_title( f"ground truth (scale={ scale }, detector window dotted)" )
    ax.figure.tight_layout()
    ax.figure.savefig( out_phantom, dpi = 150 )
    print( f"figure saved: { out_phantom }" )

    print( "reconstruction (corrected sinogram)..." )
    t0 = time.time()
    # a single `Reconstruction` for the WHOLE chain (multiscale then polish): it keeps the
    # current cloud, the trajectory (`record`) and the history from one stage to the next.
    rec = Reconstruction( corrected, extent = recon_extent, record = animate, verbose = True )

    rec.random_points( nb_diracs_final // 4**2 )
    for i in range( 3 ):
        if i:
            rec.split( factor = 4, noise_frac = 1e-2 )
        # backend = "fused": fused kernel, ~10-30x faster than the pure Jax path at this
        # scale (600 angles, thousands of Diracs -- see `benchmarks/execution_speed/benchmark_fused.py`).
        rec.diracs( backend = "fused" )
        # r = 0.08 * 2 ** ( 2 - i )
        # rec.disks( r, min_iter = disks_min_iter,
        #           disp_tol = None if disks_disp_tol_frac is None else disks_disp_tol_frac * r,
        #           split_before = disks_split_before )
        # rec.multiscale(
        #     nb_points_final = nb_diracs_final, nb_points_init = nb_diracs_final // 4**2, factor = 4,
        #     optimizer_factory = lambda n: LBFGS( max_iter = 40, ftol = 1e-9 ),
        #     noise_frac = 1e-2
        # )

        # if polish_steps > 0:
        #     print( f"polish (gradient descent + line search, { polish_steps } steps)..." )
        #     def polish_callback( step, positions ):
        #         if step % 5 == 0 or step == -1:
        #             l = rec.loss( points = positions )
        #             print( f"  step { step }: loss = { l:.6f} ({ time.time() - t0:.1f}s)" )
        #     rec.diracs( optimizer = GradientDescentLineSearch( lr = polish_lr, nb_steps = polish_steps ),
        #                 callback = polish_callback, label = "polish" )

        # rec.disks( 0.18/3, max_iter = 200, ftol = 1e-13 )

    print( f"reconstruction finished in { time.time() - t0:.1f}s" )

    rec.export_html( out_reconstruction, point_radius = point_radius,
                     title = "lung reconstruction" )


if __name__ == "__main__":
    run_two_phase_switch()
