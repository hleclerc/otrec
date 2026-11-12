"""RELOCATING THE STUCK "RED" DIRACS: a heuristic against the critical points of the semi-discrete transport.

A dirac whose mass derivative is large and positive ( red: it would rather lose mass, an isolated dirac in a pore, a
surplus in a patch ) is stuck: the gradient does not take it out. Every `every` iterations, the diracs whose mass
derivative exceeds `level` x A x spacing^2 ( an ABSOLUTE criterion: the mass derivative is a sum over the A views of dual
weights, squared lengths; calibrated on the median slice of a foam, `level` = 0.08 catches the isolated diracs and few
others ) -- and, with `still`, that moved less than `still` times the spacing over those iterations -- are moved:

- `mode = "random"`: by a Gaussian of `scale` times the spacing;
- `mode = "blue"`: next to a dirac drawn among the bottom `quantile` of the mass derivative ( blue: it would rather gain
  mass, where mass is missing ), plus a Gaussian of half a spacing.

The optimizer's memory has to be dropped after a move ( L-BFGS ).
"""
import numpy as np


class Relocation:
    def __init__( self, nb_views, level = 0.08, every = 50, still = None, mode = "blue", scale = 3.0, quantile = 0.01, seed = 0 ):
        self.nb_views, self.level, self.every, self.still, self.mode, self.scale = int( nb_views ), float( level ), int( every ), still, mode, float( scale )
        self.quantile = float( quantile )   # ( the blue pool of `mode = "blue"` )
        self.rng = np.random.default_rng( seed )
        self.anchor = None
        self.nb_moved = [ ]

    def reset( self, x ):
        """a new cloud ( a multiscale stage ): the displacements are counted from here"""
        self.anchor = np.array( x, copy = True )

    def __call__( self, it, x, mass_grad ):
        """`x` [ n, d ] after iteration `it`, its mass derivative: the moved cloud, or None when nothing moves"""
        if self.anchor is None or len( self.anchor ) != len( x ):
            self.reset( x )
            return None
        if ( it + 1 ) % self.every:
            return None
        from scipy.spatial import cKDTree
        spacing = float( np.median( cKDTree( x ).query( x, k = 2 )[ 0 ][ :, 1 ] ) )
        red = mass_grad > self.level * self.nb_views * spacing ** 2
        if self.still is not None:
            red &= np.linalg.norm( x - self.anchor, axis = 1 ) < self.still * spacing
        stuck = np.flatnonzero( red )
        self.anchor = np.array( x, copy = True )
        self.nb_moved.append( len( stuck ) )
        if len( stuck ) == 0:
            return None
        y = np.array( x, copy = True )
        if self.mode == "random":
            y[ stuck ] += self.scale * spacing * self.rng.normal( size = ( len( stuck ), x.shape[ 1 ] ) )
        else:
            lo = np.quantile( mass_grad, self.quantile )
            blue = np.flatnonzero( mass_grad <= lo )
            y[ stuck ] = x[ self.rng.choice( blue, len( stuck ) ) ] + 0.5 * spacing * self.rng.normal( size = ( len( stuck ), x.shape[ 1 ] ) )
        self.anchor[ stuck ] = y[ stuck ]
        return y
