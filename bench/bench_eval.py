"""ONE EVALUATION of the 2D reconstruction loss, value AND gradient, end to end: the number every
optimization of otrec is judged on ( an L-BFGS step is about one of them ).

    errand run --no-queue "otrec/bench/bench_eval.py::eval"                                  # here ( jax CPU )
    errand run --no-queue --env lmo-jax "otrec/bench/bench_eval.py::eval" --n=10000000       # the 2080 Ti
    errand run --no-queue "otrec/bench/bench_eval.py::eval" --method=legacy,grid --n=100000

THE PROBLEM: a sinogram of `nb_disks` random disks ( parallel beam, `angles` angles over [0, pi), `bins`
bins ), and a cloud of `n` points drawn inside those disks -- a cloud near the solution, as in the last
steps of a reconstruction.

THE METHODS ( `--method`, comma-separated ):
  legacy   `models.DiracModel.cost` under `jax.jit( jax.value_and_grad )`: the sort-based semi-discrete
           transport of sdot ( the path of `Reconstruction.diracs` before the rewrite )
  fused    `dirac_fused.diracs_cost_grad`: the CPU kernel, one thread per angle ( not jitted )
  grid     `sdot.projected1d`: deposit on a fine grid + grid transport, the new path ( `--shape`, `--k`,
           `--radius`, `--scale` )

WHAT IS TIMED: the whole call, compiled function already warm, results ready ( `block_until_ready` );
the best of `reps`. The compile time and the peak device memory ( when the device reports it ) are
printed too: we want to be FAST and PARSIMONIOUS.
"""
from errand import Param, bench


def _problem( n, angles, bins, nb_disks, seed, dtype ):
    import numpy as np
    from otrec.Sinogram import Sinogram
    rng = np.random.default_rng( seed )
    extent = 4.0
    centers = rng.uniform( -1.0, 1.0, ( nb_disks, 2 ) )
    radii = rng.uniform( 0.15, 0.5, nb_disks )
    sino = Sinogram( nb_angles = angles, nb_bins = bins, extent = extent )
    for c, r in zip( centers, radii ):
        sino.add_disk( center = c, radius = r )
    # points drawn uniformly in the union, by disk area
    which = rng.choice( nb_disks, n, p = radii ** 2 / ( radii ** 2 ).sum() )
    rho = radii[ which ] * np.sqrt( rng.random( n ) )
    phi = 2 * np.pi * rng.random( n )
    pts = centers[ which ] + rho[ :, None ] * np.stack( [ np.cos( phi ), np.sin( phi ) ], axis = 1 )
    return sino, pts.astype( dtype )


def _legacy( sino, pts ):
    import jax
    from otrec.models import DiracModel
    model = DiracModel( sino )
    return jax.jit( jax.value_and_grad( lambda q: model.cost( model.wrap( q ) ).value ) ), pts


def _fused( sino, pts ):
    from otrec.dirac_fused import diracs_cost_grad
    return ( lambda q: diracs_cost_grad( q, sino ) ), pts


def _grid( sino, pts, p ):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    loss = ProjectedLoss1d( angles = sino.angles, values = np.asarray( sino.values ), s_min = sino.s_min,
                            dw = sino.dw, scale = p.scale, shape = p.shape, k = p.k, max_radius = p.radius,
                            dtype = pts.dtype )
    from sdot.projected1d import morton_order
    radii = jnp.full( pts.shape[ 0 ], p.radius, dtype = pts.dtype )
    f = jax.jit( jax.value_and_grad( lambda q: loss( q, radii ) ) )
    # in Z order, as `Reconstruction2d` keeps them
    return f, np.asarray( pts )[ np.asarray( morton_order( pts ) ) ]


def _ready( x ):
    import jax
    return jax.tree_util.tree_map( lambda a: a.block_until_ready() if hasattr( a, "block_until_ready" ) else a, x )


def _peak_device_bytes():
    import jax
    try:
        st = jax.devices()[ 0 ].memory_stats()
        return st.get( "peak_bytes_in_use" ) if st else None
    except Exception:
        return None


if p := bench( "eval",
               n        = Param( 100000, help = "number of points ( diracs / objects )" ),
               angles   = Param( 600, help = "number of projection angles" ),
               bins     = Param( 2048, help = "detector bins" ),
               nb_disks = Param( 5, help = "disks of the phantom" ),
               method   = Param( "grid", help = "comma-separated: legacy, fused, grid" ),
               dtype    = Param( "float32", choices = [ "float32", "float64" ], help = "the points' float" ),
               shape    = Param( "dome", choices = [ "dome", "tent" ], help = "grid: the objects ( dome = density (1-|x|^2/r^2)^(k-1/2), tent = a dirac seen by the grid )" ),
               k        = Param( 1, help = "grid: the exponent of the dome ( projection (1-s^2/r^2)^k )" ),
               radius   = Param( 0.0005, help = "grid: the radius of every object" ),
               scale    = Param( 8, help = "grid: fine cells per detector bin" ),
               reps     = Param( 5, help = "repetitions ( the best is kept )" ),
               seed     = Param( 0, help = "seed of the phantom and of the cloud" ) ):
    import time
    import jax
    import numpy as np
    # float64 available ( the grid transport uses it ), the points in `--dtype`
    jax.config.update( "jax_enable_x64", True )
    print( f"device { jax.devices()[ 0 ] }, n={ p.n:.0e} angles={ p.angles } bins={ p.bins } { p.dtype }" )
    sino, pts = _problem( p.n, p.angles, p.bins, p.nb_disks, p.seed, np.dtype( p.dtype ) )
    for method in p.method.split( "," ):
        if method == "legacy":
            f, x = _legacy( sino, pts )
        elif method == "fused":
            f, x = _fused( sino, pts )
        elif method == "grid":
            f, x = _grid( sino, pts, p )
        else:
            raise ValueError( f"unknown method { method }" )
        x = jax.device_put( x )
        t0 = time.perf_counter()
        v, g = _ready( f( x ) )
        t_first = time.perf_counter() - t0
        times = [ ]
        for _ in range( p.reps ):
            t0 = time.perf_counter()
            v, g = _ready( f( x ) )
            times.append( time.perf_counter() - t0 )
        best = min( times )
        peak = _peak_device_bytes()
        gn = float( np.linalg.norm( np.asarray( g ) ) )
        print( f"  { method:8s} { best * 1e3:10.2f} ms   first call { t_first:7.2f} s   "
               f"loss { float( v ):.6e}  |grad| { gn:.3e}" + ( f"   peak { peak / 2**20:.0f} MiB" if peak else "" ) )
        p.results[ f"{ method }_ms" ] = best * 1e3
        p.results[ f"{ method }_loss" ] = float( v )


if p := bench( "parts",
               n        = Param( 1000000, help = "number of points" ),
               angles   = Param( 600, help = "number of projection angles" ),
               bins     = Param( 2048, help = "detector bins" ),
               backend  = Param( "kernel", choices = [ "kernel", "jax" ], help = "the passes over the objects" ),
               order    = Param( "random", choices = [ "random", "morton" ], help = "the order of the objects in memory" ),
               shape    = Param( "dome", choices = [ "dome", "tent" ], help = "the objects" ),
               radius   = Param( 0.0005, help = "the radius of the domes" ),
               scale    = Param( 8, help = "fine cells per detector bin" ),
               reps     = Param( 5, help = "repetitions ( the best is kept )" ) ):
    # DIAGNOSIS: the three steps of `sdot.projected1d` timed one by one ( each jitted alone, so an upper bound of
    # what they cost inside the fused evaluation )
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    sino, pts = _problem( p.n, p.angles, p.bins, 5, 0, np.float32 )
    jax.config.update( "jax_enable_x64", True )
    if p.order == "morton":
        from sdot.projected1d import morton_order
        pts = pts[ np.asarray( morton_order( pts ) ) ]
    L = ProjectedLoss1d( sino.angles, np.asarray( sino.values ), sino.s_min, sino.dw, scale = p.scale, shape = p.shape,
                         max_radius = p.radius, dtype = np.float32, backend = p.backend )
    x = jax.device_put( pts )
    r = jnp.full( p.n, p.radius, jnp.float32 )
    dep = jax.jit( L.deposit )
    mu = dep( x, r ).block_until_ready()
    if p.backend == "kernel":
        gl = jax.jit( lambda m: L._kernel_grid( m[ None ], L.geometry[ "nu" ], L.geometry[ "Cn" ], L.normals ) )
        _, g = gl( mu )
        g = g[ 0 ]
    else:
        gl = jax.jit( jax.value_and_grad( L.grid_loss ) )
        _, g = gl( mu )
        g = L.edge_differences( g )
    ga = jax.jit( L.gather )

    def best( f ):
        f()
        ts = [ ]
        for _ in range( p.reps ):
            t0 = time.perf_counter()
            jax.tree_util.tree_map( lambda a: a.block_until_ready(), f() )
            ts.append( time.perf_counter() - t0 )
        return min( ts ) * 1e3

    print( f"device { jax.devices()[ 0 ] }, n={ p.n:.0e} angles={ p.angles } bins={ p.bins } { p.backend } { p.order }" )
    for name, f in [ ( "deposit", lambda: dep( x, r ) ), ( "grid", lambda: gl( mu ) ), ( "gather", lambda: ga( x, r, g ) ) ]:
        t = best( f )
        print( f"  { name:8s} { t:10.2f} ms" )
        p.results[ name + "_ms" ] = t
