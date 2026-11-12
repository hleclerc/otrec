"""A packing of balls ( discs in 2D ) of various sizes, perfectly slippery, settled by GRAVITY in a box: a local minimum
of the potential energy  sum_i m_i z_i  under the constraint that nothing overlaps -- written as a stiff spring wherever
two balls, or a ball and a wall, overlap, the stiffness raised step by step until the overlaps are a hundredth of a radius.

The box: `|x_a| <= half_width` along the horizontal axes, a floor at `floor` along the last axis ( the vertical ), no lid."""
import numpy as np


def _energy( x, radii, masses, half_width, floor, stiffness ):
    """the energy and its gradient, `x [ n, dim ]`"""
    from scipy.spatial import cKDTree
    E, G = float( masses @ x[ :, -1 ] ), np.zeros_like( x )
    G[ :, -1 ] = masses
    # balls against balls
    pairs = cKDTree( x ).query_pairs( 2 * radii.max(), output_type = "ndarray" )
    i, j = pairs[ :, 0 ], pairs[ :, 1 ]
    d = x[ i ] - x[ j ]
    dist = np.maximum( np.linalg.norm( d, axis = 1 ), 1e-12 )
    overlap = np.maximum( radii[ i ] + radii[ j ] - dist, 0.0 )
    E += 0.5 * stiffness * float( overlap @ overlap )
    f = ( stiffness * overlap / dist )[ :, None ] * d
    np.subtract.at( G, i, f )
    np.add.at( G, j, f )
    # balls against the walls, then the floor
    side = np.maximum( np.abs( x[ :, :-1 ] ) + radii[ :, None ] - half_width, 0.0 )
    below = np.maximum( floor + radii - x[ :, -1 ], 0.0 )
    E += 0.5 * stiffness * float( ( side * side ).sum() + below @ below )
    G[ :, :-1 ] += stiffness * side * np.sign( x[ :, :-1 ] )
    G[ :, -1 ] -= stiffness * below
    return E, G


def settle( radii, dim, half_width, floor, seed = 0 ):
    """the centres `[ n, dim ]` of balls of radii `radii`, dropped from a loose random column and settled"""
    from scipy.optimize import minimize
    rng, n = np.random.default_rng( seed ), len( radii )
    masses = radii ** dim
    # the start: a column where the balls take a fifth of the room
    area = ( 2 * half_width ) ** ( dim - 1 )
    height = masses.sum() * ( np.pi if dim == 2 else 4 * np.pi / 3 ) / ( 0.2 * area )
    x = np.concatenate( [ rng.uniform( -1, 1, ( n, dim - 1 ) ) * ( half_width - radii[ :, None ] ),
                          floor + radii[ :, None ] + height * rng.random( ( n, 1 ) ) ], 1 )
    # the stiffness that leaves an overlap of a hundredth of a radius under the load of a ball of the floor
    load = masses.sum() * ( 2 * radii.mean() ) ** ( dim - 1 ) / area
    stiffness = load / ( 0.01 * radii.mean() )
    for k in ( stiffness / 100, stiffness / 10, stiffness ):
        fun = lambda v: ( lambda E, G: ( E, G.ravel() ) )( *_energy( v.reshape( n, dim ), radii, masses, half_width, floor, k ) )
        x = minimize( fun, x.ravel(), jac = True, method = "L-BFGS-B", options = dict( maxiter = 20000, maxfun = 40000, ftol = 1e-14, gtol = 1e-12 ) ).x.reshape( n, dim )
    return x


def contacts( x, radii, tol = 0.02 ):
    """the pairs `[ m, 2 ]` of balls in contact ( closer than `tol` radius to it ), and how much they overlap ( < 0: a gap )"""
    from scipy.spatial import cKDTree
    pairs = cKDTree( x ).query_pairs( 2 * radii.max() * ( 1 + tol ), output_type = "ndarray" )
    gap = radii[ pairs[ :, 0 ] ] + radii[ pairs[ :, 1 ] ] - np.linalg.norm( x[ pairs[ :, 0 ] ] - x[ pairs[ :, 1 ] ], axis = 1 )
    keep = gap > -tol * radii.mean()
    return pairs[ keep ], gap[ keep ]
