"""THE SAME 2D PROBLEM BY SEVERAL SCHEMES: the median slice ( z = 0 ) of a Voronoi foam, seen by parallel 1D projections,
reconstructed from one cloud of diracs

- (A) the 1D path of otrec ( `Reconstruction2d`: sort-free transport on a fine grid, L-BFGS on the device );
- (B) the EXACT semi-discrete 1D transport ( `semidiscrete1d`, closed form ) with the lifted Lloyd step, the scheme of the
  3D reconstruction ( one view of a 2D object is a 1D transport: no 2D solver here, which does not converge on aligned
  diracs );
- (B') the same exact transport with L-BFGS preconditioned by the Lloyd step.

The final clouds are measured the same way, by the exact transport: the cost, the MASS DERIVATIVE ( dual weights ) and
its large-scale part ( averaged over the 200 nearest diracs: the red / blue patches ), the distance to the slice.

    errand run --no-queue -o --env lmo-jax "otrec/src/otrec/experiments/exp_slice2d.py::slice2d"
"""
from errand import Param, experiment


def slice_faces( ph, nb = 512 ):
    """the slice z = 0 of `ph` as a filled raster, for the pages: its solid pixels ( occupancy >= 1/2 ) merged into
    horizontal runs, `( positions [ k, 2 ], quads [ m, 4 ] )`"""
    import numpy as np
    img, h = ph.occupancy( nb, 1 )
    solid = img[ 0 ] >= 0.5                                   # [ x, y ]
    x0 = -ph.detector / 2
    pos, quads = [ ], [ ]
    for j in range( nb ):                                     # rows of constant y
        col = np.concatenate( [ [ False ], solid[ :, j ], [ False ] ] )
        starts = np.flatnonzero( ~col[ :-1 ] & col[ 1: ] )
        ends = np.flatnonzero( col[ :-1 ] & ~col[ 1: ] )
        for i0, i1 in zip( starts, ends ):
            k = len( pos )
            xa, xb, ya, yb = x0 + i0 * h, x0 + i1 * h, x0 + j * h, x0 + ( j + 1 ) * h
            pos += [ ( xa, ya ), ( xb, ya ), ( xb, yb ), ( xa, yb ) ]
            quads.append( ( k, k + 1, k + 2, k + 3 ) )
    return np.asarray( pos ), quads


def plbfgs( evaluate, x, steps, M, memory = 10, c1 = 1e-4, max_ls = 8, record = None ):
    """L-BFGS preconditioned by the constant 2 x 2 matrix `M` ( the Lloyd step is -M grad ), Armijo from alpha = 1"""
    import numpy as np
    ev = evaluate( x )
    S, Y = [ ], [ ]
    for it in range( steps ):
        g, f = ev[ "grad" ], ev[ "cost" ]
        q, al = g.copy(), [ ]
        for s_, y_ in reversed( list( zip( S, Y ) ) ):
            a = ( s_ * q ).sum() / ( y_ * s_ ).sum()
            q -= a * y_
            al.append( a )
        gamma = ( S[ -1 ] * Y[ -1 ] ).sum() / ( Y[ -1 ] * ( Y[ -1 ] @ M.T ) ).sum() if S else 1.0
        r = gamma * ( q @ M.T )
        for ( s_, y_ ), a in zip( zip( S, Y ), reversed( al ) ):
            r += s_ * ( a - ( y_ * r ).sum() / ( y_ * s_ ).sum() )
        d = -r
        gd = ( g * d ).sum()
        if gd >= 0:
            S, Y = [ ], [ ]
            d = -( g @ M.T )
            gd = ( g * d ).sum()
        alpha = 1.0
        for _ in range( max_ls ):
            ev_t = evaluate( x + alpha * d )
            if ev_t[ "cost" ] <= f + c1 * alpha * gd:
                break
            alpha *= 0.5
        s_, y_ = alpha * d, ev_t[ "grad" ] - g
        if ( s_ * y_ ).sum() > 1e-12 * np.sqrt( ( s_ * s_ ).sum() * ( y_ * y_ ).sum() ):
            S, Y = ( S + [ s_ ] )[ -memory: ], ( Y + [ y_ ] )[ -memory: ]
        x, ev = x + alpha * d, ev_t
        if record:
            record( it, ev )
    return x


if p := experiment( "slice2d",
                    n          = Param( 5000, help = "number of diracs" ),
                    steps      = Param( 4000, help = "iterations of each method" ),
                    views      = Param( 60, help = "number of projections over [0, pi)" ),
                    bins       = Param( 256, help = "detector bins ( over a width of 2 )" ),
                    tau        = Param( 1.8, help = "(B): relaxation of the Lloyd step" ),
                    device     = Param( "cpu", choices = [ "default", "cpu" ], help = "cpu: otrec's 1D path ( A ) on the CPU, even where there is a card" ),
                    seed       = Param( 0 ) ):
    import json
    import os
    import time
    if p.device == "cpu":   # before loom and jax are imported
        from otrec.device import use_cpu
        use_cpu()

    import numpy as np
    from scipy.spatial import cKDTree

    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d
    from otrec.semidiscrete1d import Views1d

    # the slice of the thin foam ( the same walls as the 0.2 slab )
    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    views = Views1d( sino.normals, np.asarray( sino.values ), sino.s_min, sino.dw )

    # the start: uniform in the disk
    rng = np.random.default_rng( p.seed )
    rho, phi = 0.8 * np.sqrt( rng.random( p.n ) ), 2 * np.pi * rng.random( p.n )
    x0 = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ) ], 1 )

    ref = ph.sample( 40000, seed = 9 )[ :, :2 ]
    tol = E / B

    def measure( name, x, wall, costs ):
        ev = views.evaluate( x )
        mg = ev[ "mass_grad" ]
        large = mg[ cKDTree( x ).query( x, k = 200 )[ 1 ] ].mean( 1 )
        d = ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )
        res = dict( method = name, cost = ev[ "cost" ], mass_grad_std = float( mg.std() ), large_scale_std = float( large.std() ),
                    large_ratio = float( large.std() / mg.std() ), inside = float( ( d <= tol ).mean() ), dist_mean = float( d.mean() ),
                    coverage = float( cKDTree( x ).query( ref )[ 0 ].mean() ), wall = wall )
        print( "  ".join( f"{ k } { v:.4e}" if isinstance( v, float ) else f"{ k } { v }" for k, v in res.items() ), flush = True )
        return res, dict( x = x, mass = mg, costs = np.asarray( costs ) )

    results, finals = [ ], { }

    # (A) otrec's 1D path
    t0 = time.perf_counter()
    ra = Reconstruction2d( sino, x0, extent = E )
    ra.diracs( max_iter = p.steps, ftol = 0.0, min_iter = p.steps )
    r, f = measure( "A", np.asarray( ra.positions ), time.perf_counter() - t0, ra.history[ -1 ][ "losses" ] )
    results.append( r ); finals[ "A" ] = f

    # (B) exact 1D transport + lifted Lloyd
    t0 = time.perf_counter()
    x, costs = x0.copy(), [ ]
    for it in range( p.steps ):
        ev = views.evaluate( x )
        costs.append( ev[ "cost" ] )
        x = x + p.tau * ev[ "step" ]
        if it % 500 == 0:
            print( f"  B { it:5d}  cost { ev[ 'cost' ]:.6e}", flush = True )
    r, f = measure( "B", x, time.perf_counter() - t0, costs )
    results.append( r ); finals[ "B" ] = f

    # (B') exact 1D transport + preconditioned L-BFGS
    t0 = time.perf_counter()
    costs = [ ]

    def rec( it, ev ):
        costs.append( ev[ "cost" ] )
        if it % 500 == 0:
            print( f"  B' { it:5d}  cost { ev[ 'cost' ]:.6e}", flush = True )

    M = 0.5 * p.n * np.linalg.inv( views.H )
    x = plbfgs( views.evaluate, x0.copy(), p.steps, M, record = rec )
    r, f = measure( "Bp", x, time.perf_counter() - t0, costs )
    results.append( r ); finals[ "Bp" ] = f

    out = p.out_dir
    np.savez_compressed( out / "slice2d.npz", ref = ref, **{ f"{ k }_{ q }": v for k, f in finals.items() for q, v in f.items() } )
    ( out / "results.json" ).write_text( json.dumps( results, indent = 1 ) )

    # a page: the slice, then the final clouds colored by their mass derivative ( blue -> white -> red, each on its own
    # scale )
    from sdot.viz.Visualizer import Visualizer
    from otrec.viz.view3d import _BWR, _edges
    v = Visualizer( title = "slice2d", frame_axis = "method", frame_value = 0, fps = 1.0 )
    v.add_faces( *slice_faces( ph ), color = ( "#7a7a7a", "#8a8a8a" ) )
    for i, k in enumerate( finals ):
        v.new_frame( i + 1 )
        x, mg = finals[ k ][ "x" ], finals[ k ][ "mass" ]
        cls = np.searchsorted( _edges( mg, True ), mg )
        for c, col in enumerate( _BWR ):
            m = cls == c
            if m.any():
                v.add_points( x[ m ], color = col )
    v.write_html( str( out / "slice2d.html" ) )
    for r in results:
        p.results[ r[ "method" ] + "_large_ratio" ] = r[ "large_ratio" ]


if p := experiment( "escape",
                    frozen     = Param( "", help = "the slice2d.npz whose ( B ) and ( A ) clouds to start from ( no comma in the path )" ),
                    steps      = Param( 2000, help = "iterations of each continuation" ),
                    noise      = Param( "0.1:0.3:1.0", help = "':'-separated perturbations, in units of the mean spacing" ),
                    views      = Param( 60 ),
                    bins       = Param( 256 ),
                    tau        = Param( 1.8, help = "relaxation of the Lloyd step" ),
                    device     = Param( "cpu", choices = [ "default", "cpu" ] ),
                    seed       = Param( 0 ) ):
    # FROM THE FROZEN POINT of the exact Lloyd ( B ): is it a trap of the exact landscape? ( 1 ) perturbed, then exact
    # Lloyd; ( 2 ) the smoothed transport ( otrec's 1D path: tents of width h ) from it; ( 3 ) the control, exact Lloyd
    # from the 1D path's point ( A ). Every result measured by the exact transport.
    import json
    import os
    if p.device == "cpu":
        from otrec.device import use_cpu
        use_cpu()
    import numpy as np
    from scipy.spatial import cKDTree
    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d
    from otrec.semidiscrete1d import Views1d

    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    views = Views1d( sino.normals, np.asarray( sino.values ), sino.s_min, sino.dw )
    d = np.load( p.frozen )
    xb, xa = d[ "B_x" ], d[ "A_x" ]
    n = len( xb )
    spacing = float( cKDTree( xb ).query( xb, k = 2 )[ 0 ][ :, 1 ].mean() )

    def inside( x ):
        dist = ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )
        return float( ( dist <= E / B ).mean() )

    def lloyd( x, steps ):
        costs = [ ]
        for _ in range( steps ):
            ev = views.evaluate( x )
            costs.append( ev[ "cost" ] )
            x = x + p.tau * ev[ "step" ]
        return x, costs

    results = [ ]

    def report( name, x, costs ):
        c = views.evaluate( x )[ "cost" ]
        r = dict( name = name, cost = c, first = float( costs[ 0 ] ) if len( costs ) else c, inside = inside( x ),
                  frozen_after = int( np.argmax( np.abs( np.diff( costs ) ) < 1e-12 * c ) ) if len( costs ) > 1 else 0 )
        print( "  ".join( f"{ k } { v:.4e}" if isinstance( v, float ) else f"{ k } { v }" for k, v in r.items() ), flush = True )
        results.append( r )

    report( "B frozen ( start )", xb, [ ] )
    report( "A ( 1D path )", xa, [ ] )
    rng = np.random.default_rng( p.seed )
    for s in [ float( v ) for v in p.noise.split( ":" ) ]:
        x0 = xb + s * spacing * rng.normal( size = xb.shape )
        x, costs = lloyd( x0, p.steps )
        report( f"B + noise { s } spacing, then exact Lloyd", x, costs )
    # the smoothed transport from B's point
    ra = Reconstruction2d( sino, xb, extent = E )
    ra.diracs( max_iter = p.steps, ftol = 0.0, min_iter = p.steps )
    xs = np.asarray( ra.positions )
    report( "B, then the 1D path ( tents of width h, L-BFGS )", xs, ra.history[ -1 ][ "losses" ] )
    x, costs = lloyd( xs, p.steps )
    report( "B, then the 1D path, then exact Lloyd", x, costs )
    # the control
    x, costs = lloyd( xa, p.steps )
    report( "A, then exact Lloyd", x, costs )
    ( p.out_dir / "escape.json" ).write_text( json.dumps( results, indent = 1 ) )


if p := experiment( "anim1d",
                    n          = Param( 5000, help = "number of diracs" ),
                    steps      = Param( 4000, help = "L-BFGS iterations" ),
                    record     = Param( 50, help = "a frame every that many iterations" ),
                    views      = Param( 60, help = "number of projections over [0, pi)" ),
                    bins       = Param( 256, help = "detector bins ( over a width of 2 )" ),
                    device     = Param( "cpu", choices = [ "default", "cpu" ] ),
                    seed       = Param( 0 ) ):
    # THE CONVERGENCE OF otrec's 1D PATH on the median slice, frame by frame: the diracs colored by their distance to the
    # slice ( blue: within 1 bin, white: 1 to 2 bins, red: farther ), the slice itself as the first frame, and the curves
    # ( the loss, the fraction of diracs farther than 0.5, 1, 2 bins ).
    import json
    import os
    if p.device == "cpu":
        from otrec.device import use_cpu
        use_cpu()
    import numpy as np
    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d
    from otrec.semidiscrete1d import Views1d
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.convergence import write_convergence_html

    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    views = Views1d( sino.normals, np.asarray( sino.values ), sino.s_min, sino.dw )
    rng = np.random.default_rng( p.seed )
    rho, phi = 0.8 * np.sqrt( rng.random( p.n ) ), 2 * np.pi * rng.random( p.n )
    x0 = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ) ], 1 )

    rec = Reconstruction2d( sino, x0, extent = E, record_every = p.record )
    rec.diracs( max_iter = p.steps, ftol = 0.0, min_iter = p.steps )
    frames = rec.frames
    bin_ = E / B

    def dist( x ):
        return ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )

    v = Visualizer( title = f"1D path, { A } projections", frame_axis = "iteration", frame_value = -1, fps = 6.0 )
    v.add_faces( *slice_faces( ph ), color = ( "#7a7a7a", "#8a8a8a" ) )
    curves = { "outside 0.5 bin": [ ], "outside 1 bin": [ ], "outside 2 bins": [ ], "exact cost": [ ] }
    for i, x in enumerate( frames ):
        it = i * p.record
        v.new_frame( it )
        d = dist( x )
        for mask, col in ( ( d <= bin_, ( "#2166ac", "#6fb0e0" ) ), ( ( d > bin_ ) & ( d <= 2 * bin_ ), ( "#9a9a9a", "#f2f2f2" ) ),
                           ( d > 2 * bin_, ( "#b2182b", "#e0303f" ) ) ):
            if mask.any():
                v.add_points( x[ mask ], color = col )
        for k, t in ( ( "outside 0.5 bin", 0.5 ), ( "outside 1 bin", 1 ), ( "outside 2 bins", 2 ) ):
            curves[ k ].append( ( it, float( ( d > t * bin_ ).mean() ) ) )
        curves[ "exact cost" ].append( ( it, views.evaluate( np.asarray( x, dtype = float ) )[ "cost" ] ) )
    out = p.out_dir
    v.write_html( str( out / "animation.html" ) )
    write_convergence_html( curves, out / "curves.html", title = f"1D path, { A } projections", xlabel = "iteration", ylabel = "" )
    last = { k: c[ -1 ][ 1 ] for k, c in curves.items() }
    print( f"views { A }: " + "  ".join( f"{ k } { v:.4e}" for k, v in last.items() ), flush = True )
    ( out / "last.json" ).write_text( json.dumps( last, indent = 1 ) )
    np.savez_compressed( out / "anim1d.npz", frames = np.stack( frames ).astype( np.float32 ) )
    for k, val in last.items():
        p.results[ k.replace( " ", "_" ) ] = val


if p := experiment( "relocate",
                    n          = Param( 5000, help = "number of diracs at the end" ),
                    n_init     = Param( 1250, help = "multiscale: the first stage" ),
                    stage_steps = Param( "1500:1500", help = "':'-separated iterations of each stage" ),
                    views      = Param( 600 ),
                    bins       = Param( 256 ),
                    tau        = Param( 1.8, help = "relaxation of the Lloyd step" ),
                    mode       = Param( "none:random:blue", help = "':'-separated variants: none, random, blue" ),
                    every      = Param( 50 ),
                    level      = Param( 0.08, help = "relocate when the mass derivative exceeds level x views x spacing^2" ),
                    settle     = Param( 300, help = "no relocation in the last that many iterations of a stage" ),
                    warmup     = Param( 300, help = "no relocation in the first that many iterations of a stage ( it is converging )" ),
                    scale      = Param( 3.0 ),
                    record     = Param( 100, help = "a frame every that many iterations" ),
                    seed       = Param( 0 ) ):
    # THE RED-DIRAC HEURISTIC on the median slice: multiscale, exact 1D transport, exact Lloyd, with and without
    # `relocation.Relocation`; measured by the exact cost and the distance to the slice; an animation per variant.
    import json
    import numpy as np
    from scipy.spatial import cKDTree
    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.semidiscrete1d import Views1d
    from otrec.relocation import Relocation
    from sdot.viz.Visualizer import Visualizer
    from otrec.viz.view3d import _BWR, _edges

    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    views = Views1d( sino.normals, np.asarray( sino.values ), sino.s_min, sino.dw )
    bin_ = E / B
    ref = ph.sample( 40000, seed = 9 )[ :, :2 ]
    steps = [ int( s ) for s in p.stage_steps.split( ":" ) ]

    def dist( x ):
        return ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )

    results = [ ]
    for mode in p.mode.split( ":" ):
        rng = np.random.default_rng( p.seed )
        rho, phi = 0.8 * np.sqrt( rng.random( p.n_init ) ), 2 * np.pi * rng.random( p.n_init )
        x = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ) ], 1 )
        rel = Relocation( A, level = p.level, every = p.every, mode = mode, scale = p.scale, seed = p.seed ) if mode != "none" else None
        v = Visualizer( title = f"relocation { mode }", frame_axis = "iteration", frame_value = -1, fps = 6.0 )
        v.add_faces( *slice_faces( ph ), color = ( "#7a7a7a", "#8a8a8a" ) )
        it_total, costs = 0, [ ]
        for stage, nb in enumerate( steps ):
            if stage:   # split
                sp = float( np.median( cKDTree( x ).query( x, k = 2 )[ 0 ][ :, 1 ] ) )
                x = np.repeat( x, 4, axis = 0 ) + 0.25 * sp * rng.normal( size = ( 4 * len( x ), 2 ) )
                x = x[ rng.choice( len( x ), min( len( x ), p.n ), replace = False ) ]
            if rel is not None:
                rel.reset( x )
            for it in range( nb ):
                ev = views.evaluate( x )
                costs.append( ev[ "cost" ] )
                if it_total % p.record == 0:
                    v.new_frame( it_total )
                    mg = ev[ "mass_grad" ]
                    cls = np.searchsorted( _edges( mg, True ), mg )
                    for c, col in enumerate( _BWR ):
                        m = cls == c
                        if m.any():
                            v.add_points( x[ m ], color = col )
                x = x + p.tau * ev[ "step" ]
                if rel is not None and p.warmup <= it < nb - p.settle:
                    y = rel( it, x, ev[ "mass_grad" ] )
                    if y is not None:
                        x = y
                it_total += 1
        d = dist( x )
        r = dict( mode = mode, cost = views.evaluate( x )[ "cost" ], out_05 = float( ( d > 0.5 * bin_ ).mean() ),
                  out_1 = float( ( d > bin_ ).mean() ), out_2 = float( ( d > 2 * bin_ ).mean() ),
                  moved = int( sum( rel.nb_moved ) ) if rel else 0 )
        print( "  ".join( f"{ k } { val:.4e}" if isinstance( val, float ) else f"{ k } { val }" for k, val in r.items() ), flush = True )
        results.append( r )
        v.write_html( str( p.out_dir / f"relocation_{ mode }.html" ) )
        np.savez_compressed( p.out_dir / f"relocation_{ mode }.npz", x = x, costs = np.asarray( costs ) )
    ( p.out_dir / "relocate.json" ).write_text( json.dumps( results, indent = 1 ) )


if p := experiment( "tents",
                    n          = Param( 5000, help = "number of diracs at the end" ),
                    n_init     = Param( 1250, help = "multiscale: the first stage" ),
                    stage_steps = Param( "1500:1500", help = "':'-separated iterations of each stage" ),
                    views      = Param( 60 ),
                    bins       = Param( 256 ),
                    tau        = Param( 1.8, help = "relaxation of the Lloyd step" ),
                    widths     = Param( "0:0.125:0.5:1", help = "':'-separated tent half-widths, in detector bins ( 0: exact diracs )" ),
                    optimizer  = Param( "lloyd", help = "':'-separated: lloyd ( at tau ), plbfgs ( L-BFGS preconditioned by the Lloyd step )" ),
                    memory     = Param( 10, help = "plbfgs: the number of pairs kept" ),
                    record     = Param( 100, help = "a frame every that many iterations" ),
                    device     = Param( "cpu", choices = [ "default", "cpu" ], help = "cpu: jax ( the slice's projection ) on the host" ),
                    seed       = Param( 0 ) ):
    import os
    if p.device == "cpu":   # before jax is imported
        from otrec.device import use_cpu
        use_cpu()
    # TENTS INSTEAD OF DIRACS in the exact 1D transport ( `semidiscrete1d.transport_tents`, W2 between the measured
    # profile and a SumOfTents ): multiscale exact Lloyd for each half-width, measured by the distance to the slice and by
    # the cost with exact diracs; an animation per width ( colored by the distance to the slice ).
    import json
    import numpy as np
    from scipy.spatial import cKDTree
    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.semidiscrete1d import Views1d
    from sdot.viz.Visualizer import Visualizer

    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    views = Views1d( sino.normals, np.asarray( sino.values ), sino.s_min, sino.dw )
    bin_ = E / B
    ref = ph.sample( 40000, seed = 9 )[ :, :2 ]
    steps = [ int( s ) for s in p.stage_steps.split( ":" ) ]

    def dist( x ):
        return ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )

    results = [ ]
    for opt, wb in [ ( o, float( v ) ) for o in p.optimizer.split( ":" ) for v in p.widths.split( ":" ) ]:
        w = wb * bin_
        rng = np.random.default_rng( p.seed )
        rho, phi = 0.8 * np.sqrt( rng.random( p.n_init ) ), 2 * np.pi * rng.random( p.n_init )
        x = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ) ], 1 )
        v = Visualizer( title = f"tents { wb } bin, { opt }", frame_axis = "iteration", frame_value = -1, fps = 6.0 )
        v.add_faces( *slice_faces( ph ), color = ( "#7a7a7a", "#8a8a8a" ) )
        it_total, costs = 0, [ ]
        for stage, nb in enumerate( steps ):
            if stage:
                sp = float( np.median( cKDTree( x ).query( x, k = 2 )[ 0 ][ :, 1 ] ) )
                x = np.repeat( x, 4, axis = 0 ) + 0.25 * sp * rng.normal( size = ( 4 * len( x ), 2 ) )
                x = x[ rng.choice( len( x ), min( len( x ), p.n ), replace = False ) ]
            def frame( it_total, x ):
                v.new_frame( it_total )
                d = dist( x )
                for mask, col in ( ( d <= bin_, ( "#2166ac", "#6fb0e0" ) ), ( ( d > bin_ ) & ( d <= 2 * bin_ ), ( "#9a9a9a", "#f2f2f2" ) ),
                                   ( d > 2 * bin_, ( "#b2182b", "#e0303f" ) ) ):
                    if mask.any():
                        v.add_points( x[ mask ], color = col )

            if opt == "plbfgs":
                state = dict( x = x )
                start = it_total

                def evaluate( y ):
                    state[ "x" ] = y
                    return views.evaluate( y, tent = w )

                def rec( it, ev ):
                    costs.append( ev[ "cost" ] )
                    if ( start + it ) % p.record == 0:
                        frame( start + it, state[ "x" ] )

                x = plbfgs( evaluate, x, nb, 0.5 * len( x ) * np.linalg.inv( views.H ), memory = p.memory, record = rec )
                it_total += nb
            else:
                for it in range( nb ):
                    ev = views.evaluate( x, tent = w )
                    costs.append( ev[ "cost" ] )
                    if it_total % p.record == 0:
                        frame( it_total, x )
                    x = x + p.tau * ev[ "step" ]
                    it_total += 1
        d = dist( x )
        r = dict( optimizer = opt, width_bins = wb, cost_tents = costs[ -1 ], cost_diracs = views.evaluate( x )[ "cost" ], out_05 = float( ( d > 0.5 * bin_ ).mean() ),
                  out_1 = float( ( d > bin_ ).mean() ), out_2 = float( ( d > 2 * bin_ ).mean() ),
                  last_change = float( abs( costs[ -1 ] - costs[ -min( 200, len( costs ) ) ] ) / costs[ -1 ] ) )
        print( "  ".join( f"{ k } { val:.4e}" if isinstance( val, float ) else f"{ k } { val }" for k, val in r.items() ), flush = True )
        results.append( r )
        v.write_html( str( p.out_dir / f"tents_{ opt }_{ wb }.html" ) )
        np.savez_compressed( p.out_dir / f"tents_{ opt }_{ wb }.npz", x = x, costs = np.asarray( costs ) )
    ( p.out_dir / "tents.json" ).write_text( json.dumps( results, indent = 1 ) )


if p := experiment( "tents2d",
                    n          = Param( 5000, help = "number of diracs at the end" ),
                    n_init     = Param( 1250, help = "multiscale: the first stage" ),
                    max_iter   = Param( 100, help = "L-BFGS iterations per stage" ),
                    views      = Param( 600 ),
                    bins       = Param( 256 ),
                    width      = Param( 0.5, help = "tent half-width, in detector bins" ),
                    record     = Param( 10, help = "a frame every that many iterations" ),
                    device     = Param( "cpu", choices = [ "default", "cpu" ] ),
                    method     = Param( "exact", choices = [ "exact", "grid" ], help = "the tents' transport: exact, or on a fine grid" ),
                    scale      = Param( 8, help = "grid method: fine cells per bin" ),
                    seed       = Param( 0 ) ):
    # otrec's Reconstruction2d with EXACT tents ( sdot's ProjectedSumOfTents + ContinuousOtPlan1d ), multiscale, the
    # device L-BFGS: the distance to the slice, the time, an animation.
    import json
    import os
    import time
    if p.device == "cpu":
        from otrec.device import use_cpu
        use_cpu()
    import numpy as np
    from otrec import phantoms
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d
    from sdot.viz.Visualizer import Visualizer

    nb_cells = int( round( np.pi * 0.8 ** 2 * 0.2 / 0.19 ** 3 ) )
    ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = 0.8, height = 0.02, seed = p.seed, seed_height = 0.2 )
    A, B, E = p.views, p.bins, 2.0
    sino = Sinogram( nb_angles = A, nb_bins = B, extent = E )
    sino.values = ph.slice_projections( sino.normals, sino.s_min, sino.dw, B )
    bin_ = E / B
    rec = Reconstruction2d( sino, extent = 1.6, max_iter = p.max_iter, ftol = 0.0, min_iter = p.max_iter,
                            record_every = p.record, verbose = True, dtype = np.float64, seed = p.seed, scale = int( p.scale ) )
    t0 = time.perf_counter()
    rec.multiscale( p.n, nb_points_init = p.n_init, factor = 4, shape = "tents_grid" if p.method == "grid" else "tents",
                    radius = p.width * bin_ )
    wall = time.perf_counter() - t0

    def dist( x ):
        return ph.distance_to_support( np.concatenate( [ x, np.zeros( ( len( x ), 1 ) ) ], 1 ) )

    d = dist( rec.positions )
    r = dict( out_05 = float( ( d > 0.5 * bin_ ).mean() ), out_1 = float( ( d > bin_ ).mean() ), out_2 = float( ( d > 2 * bin_ ).mean() ),
              wall = wall, evals = int( sum( h[ "nb_evals" ] for h in rec.history ) ), loss = rec.history[ -1 ][ "loss_after" ] )
    print( "  ".join( f"{ k } { v:.4e}" if isinstance( v, float ) else f"{ k } { v }" for k, v in r.items() ), flush = True )
    ( p.out_dir / "tents2d.json" ).write_text( json.dumps( r, indent = 1 ) )
    v = Visualizer( title = "exact tents, Reconstruction2d", frame_axis = "frame", frame_value = -1, fps = 6.0 )
    v.add_faces( *slice_faces( ph ), color = ( "#7a7a7a", "#8a8a8a" ) )
    for i, x in enumerate( rec.frames ):
        v.new_frame( i )
        d = dist( np.asarray( x ) )
        for mask, col in ( ( d <= bin_, ( "#2166ac", "#6fb0e0" ) ), ( ( d > bin_ ) & ( d <= 2 * bin_ ), ( "#9a9a9a", "#f2f2f2" ) ),
                           ( d > 2 * bin_, ( "#b2182b", "#e0303f" ) ) ):
            if mask.any():
                v.add_points( np.asarray( x )[ mask ], color = col )
    v.write_html( str( p.out_dir / "tents2d.html" ) )
    for k, val in r.items():
        p.results[ k ] = val
