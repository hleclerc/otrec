"""Steps to try in a STRATEGY ( world 2D, balls ), each recorded as a step of the `Reconstruction`, so that an experiment
calls them between its stages:

    relax_spacing   the balls moved towards a regular spacing: a few steps of Lloyd restricted to the union of their
                    discs ( each ball to the centroid of its Voronoi cell cut by a disc of `reach` radii: a quantization
                    of the matter the balls draw, without the cloud shrinking at its borders )
    coarsen_interior  the balls deep in the matter ( what the balls draw ) merged 4 by 4 into balls twice as big
    refine_borders  the balls of the borders the views contest the most ( a recess they cannot follow ) split in 4
                    ( the cloud then has several radii: `adjust_parameters( radii = "keep" )` moves its centres )"""
import math
import time

import numpy as np


def _record( rec, step, **info ):
    """a step of `rec` that changed its cloud: its loss, a frame"""
    lf, x = rec._rows()
    loss = lf.value_and_grad( x, 0.0, rec.background )[ 0 ]
    rec.recorder.step( step, snapshot = lambda: rec._snapshot( lf, x, 0.0, rec.background ), nb_points = rec.nb_points, radius = rec.radius,
                       loss = loss, **info )


def relax_spacing( rec, iterations = 2, reach = 1.5, samples = 128, smallest_only = True ):
    """`iterations` steps of Lloyd restricted to the union of the discs of radius `reach r` around the balls ( `r` the
    radius of each ): each ball to the centroid of the samples of its power cell ( weight r^2: balls of several sizes )
    ( `samples` per ball, evenly spread in its disc: a sunflower ). `smallest_only`: only the balls of the smallest
    radius move, the others are sites that stay ( Lloyd among balls of several sizes takes the fit apart )"""
    from scipy.spatial import cKDTree
    t0, r = time.perf_counter(), rec.radii
    P = rec.points.copy()
    n = len( P )
    k = np.arange( samples ) + 0.5                                              # a sunflower: the same even samples for every ball
    th = math.pi * ( 3 - math.sqrt( 5 ) ) * k
    off = ( reach * r )[ :, None, None ] * ( np.sqrt( k / samples )[ :, None ] * np.stack( [ np.cos( th ), np.sin( th ) ], 1 ) )[ None ]
    move0 = None
    for _ in range( iterations ):
        S = ( P[ :, None, : ] + off ).reshape( -1, 2 )
        # the POWER cells ( weights r^2: a big ball keeps its share among small ones ): |x - p|^2 - r^2, the distance
        # in 3D to ( p, sqrt( M - r^2 ) ) less M, for the samples at height 0
        M = float( ( r ** 2 ).max() )
        lift = np.concatenate( [ P, np.sqrt( M - r ** 2 )[ :, None ] ], 1 )
        _, j = cKDTree( lift ).query( np.concatenate( [ S, np.zeros( ( len( S ), 1 ) ) ], 1 ), workers = -1 )
        ok = np.linalg.norm( S - P[ j ], axis = 1 ) < reach * r[ j ]
        cnt = np.bincount( j[ ok ], minlength = n )
        c = np.stack( [ np.bincount( j[ ok ], S[ ok, k ], n ) for k in ( 0, 1 ) ], 1 )
        new = np.where( cnt[ :, None ] > 0, c / np.maximum( cnt, 1 )[ :, None ], P )
        if smallest_only:
            new = np.where( ( r <= r.min() * ( 1 + 1e-9 ) )[ :, None ], new, P )
        move0 = np.mean( np.linalg.norm( new - P, axis = 1 ) / r ) if move0 is None else move0
        P = new
    rec._set( rec._inside( P ), r, rec.densities )
    _record( rec, "relax_spacing", iterations = iterations, reach = reach, mean_move = float( move0 ), seconds = time.perf_counter() - t0 )
    return rec


def matter_mask( rec, area, nb = 1024, level = 0.5, smooth = 1.0 ):
    """`( mask [ nb, nb ], pixel )`: where what the balls draw is matter ( their density, smoothed over `smooth` times
    the median radius -- between two bumps it dips --, above `level` of the white; `area`: of the matter,
    `Sinogram.matter_area` )"""
    from scipy import ndimage as nd
    from otrec.shapes import sized
    md, E = rec.model, rec.sinogram.extent
    pix = E / nb
    img = sized.density( md.tensor( rec.points ), md.tensor( rec.radii ), nb, E, md.k ) * area / pix ** 2
    img = nd.gaussian_filter( img, smooth * float( np.median( rec.radii ) ) / pix )
    return img > level, pix


def depth( rec, mask, pix ):
    """`[ n ]`: how deep in the matter each ball is: the distance from its centre to the void"""
    from scipy import ndimage as nd
    d = nd.distance_transform_edt( mask ) * pix
    E = rec.sinogram.extent
    ij = np.clip( ( ( rec.points + E / 2 ) / pix ).astype( int ), 0, mask.shape[ 0 ] - 1 )
    return d[ ij[ :, 0 ], ij[ :, 1 ] ]


def coarsen_interior( rec, area, room = 1.2, passes = 3, max_ratio = None, next_division = 4 ):
    """the balls DEEP in the matter merged 4 by 4: a ball and its 3 nearest of the same radius, all deeper than `room`
    times the radius they would make ( twice theirs ), become one ball of that radius at their centroid ( the same
    mass ). `passes` times ( a ball merged may merge again ), the radii staying within `max_ratio` of the smallest
    radius once the borders are split in `next_division` ( `refine_borders`: then `rec.radius / sqrt( next_division )` )"""
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    mask, pix = matter_mask( rec, area )
    top = ( max_ratio or rec.model.max_ratio ) * rec.radius / math.sqrt( next_division )
    merged = 0
    for _ in range( passes ):
        P, r = rec.points, rec.radii
        dp = depth( rec, mask, pix )
        cand = np.nonzero( ( dp > room * 2 * r ) & ( 2 * r <= 0.95 * top ) )[ 0 ]          # ( not on the bound of the kernels )
        if len( cand ) < 4:
            break
        used = np.zeros( len( P ), bool )
        tree = cKDTree( P[ cand ] )
        new_p, new_r = [ ], [ ]
        for i in cand[ np.argsort( -dp[ cand ] ) ]:                              # the deepest first
            if used[ i ]:
                continue
            _, jj = tree.query( P[ i ], min( 8, len( cand ) ) )
            group = [ i ] + [ cand[ j ] for j in np.atleast_1d( jj ) if j < len( cand ) and cand[ j ] != i and not used[ cand[ j ] ] and r[ cand[ j ] ] == r[ i ] ][ :3 ]
            if len( group ) < 4 or np.linalg.norm( P[ group ] - P[ i ], axis = 1 ).max() > 4 * r[ i ]:
                continue
            used[ group ] = True
            new_p.append( P[ group ].mean( 0 ) ); new_r.append( 2 * r[ i ] )
        if not new_p:
            break
        merged += len( new_p )
        rec._set( np.concatenate( [ P[ ~used ], np.array( new_p ) ] ), np.concatenate( [ r[ ~used ], np.array( new_r ) ] ) )
    _record( rec, "coarsen_interior", merged = merged, seconds = time.perf_counter() - t0 )
    return rec


def refine_borders( rec, area, share = 0.5, room = 1.2, division = 4, noise_ratio = 0.5 ):
    """the balls of the BORDERS ( less deep in the matter than `room` radii ) that the views contest the most ( their
    `dissonance`, `Reconstruction.fields`: where a ball does not fit the data, a recess it cannot follow ), the `share`
    of them, each in `division` ( radius / sqrt( division ) ); `rec.radius` becomes the smallest radius"""
    t0 = time.perf_counter()
    mask, pix = matter_mask( rec, area )
    P, r = rec.points, rec.radii
    border = depth( rec, mask, pix ) < room * r
    dis = rec.fields()[ "dissonance" ]
    split = np.zeros( len( P ), bool )
    b = np.nonzero( border )[ 0 ]
    split[ b[ np.argsort( -dis[ b ] )[ : int( share * len( b ) ) ] ] ] = True
    kids = np.repeat( P[ split ], division, axis = 0 )
    rk = np.repeat( r[ split ], division ) / math.sqrt( division )
    kids += 0.5 * noise_ratio * rk[ :, None ] * rec.rng.normal( size = kids.shape )
    radii = np.concatenate( [ r[ ~split ], rk ] )
    rec.model.radius = float( radii.min() )
    rec._set( rec._inside( np.concatenate( [ P[ ~split ], kids ] ) ), radii )
    _record( rec, "refine_borders", border = int( border.sum() ), split = int( split.sum() ), seconds = time.perf_counter() - t0 )
    return rec
