"""The measured data: one image of the detector per view, for a 2D world ( 1D detector ) or a 3D one ( 2D detector ).

    values[ view, i ]       world 2D: `nb_bins` bins
    values[ view, i, j ]    world 3D: `nb_bins x nb_bins` pixels

`values` holds the INTENSITY ( the mass ) of each pixel. The detector is centred, of side `extent`, the same in every view;
`axes[ view, a, : ]` is the unit vector of the world along the detector axis `a`: a point `p` projects at `axes[ view ] @ p`.

    world 2D   axes = ( cos t, sin t ),                       t = view pi / nb_views
    world 3D   axes = ( cos t, sin t, 0 ), ( 0, 0, 1 )        ( rotation about z ), or any `axes` given

`diracs( n )` gives the measure the reconstruction transports: `n` diracs per pixel and per axis, on a regular grid, each
carrying its share of the intensity of its pixel."""
import numpy as np

from .projection import project_image
from .shapes import gaussian_matrix


class Sinogram:
    def __init__( self, dim, nb_views, nb_bins, extent, axes = None, values = None ):
        if dim not in ( 2, 3 ):
            raise ValueError( "Sinogram: dim must be 2 or 3" )
        self.dim, self.nb_views, self.nb_bins = int( dim ), int( nb_views ), int( nb_bins )
        self.extent = float( extent )
        self.h = self.extent / self.nb_bins
        self.origin = -self.extent / 2
        self.axes = self.default_axes( self.dim, self.nb_views ) if axes is None else np.asarray( axes, dtype = float )
        if self.axes.shape != ( self.nb_views, self.dim - 1, self.dim ):
            raise ValueError( f"Sinogram: axes must have shape { ( self.nb_views, self.dim - 1, self.dim ) }" )
        shape = ( self.nb_views, ) + ( self.nb_bins, ) * ( self.dim - 1 )
        self.values = np.zeros( shape ) if values is None else np.array( values, dtype = float ).reshape( shape )

    @staticmethod
    def default_axes( dim, nb_views ):
        t = np.pi * np.arange( nb_views ) / nb_views
        u = np.stack( [ np.cos( t ), np.sin( t ) ] + [ np.zeros_like( t ) ] * ( dim - 2 ), -1 )
        return u[ :, None, : ] if dim == 2 else np.stack( [ u, np.broadcast_to( [ 0.0, 0.0, 1.0 ], u.shape ) ], 1 )

    @classmethod
    def from_image( cls, image, nb_views, nb_bins, extent = 2.0 ):
        """the sinogram of an IMAGE `[ nx, nx ]` ( world 2D ) or `[ nx, nx, nz ]` ( world 3D ) given on the centred square or
        cube of side `extent`: the integrals along the rays ( `projection` ), for views that turn about z"""
        s = cls( np.ndim( image ), nb_views, nb_bins, extent )
        s.values = project_image( image, extent, s.axes[ :, 0, :2 ], nb_bins )
        return s

    # -- geometry ----------------------------------------------------------------------------------------------------------

    def project( self, points ):
        """`[ nb_views, n, dim - 1 ]`: the detector coordinates of `points [ n, dim ]`"""
        return ( self.axes @ points.T ).transpose( 0, 2, 1 )

    @property
    def edges( self ):
        return self.origin + self.h * np.arange( self.nb_bins + 1 )

    # -- the transported measure -------------------------------------------------------------------------------------------

    def blurred( self, sigma ):
        """`values` blurred by a gaussian of width `sigma` PIXELS along each detector axis ( a copy )"""
        G = gaussian_matrix( self.nb_bins, sigma )
        v = np.einsum( "ai,vi...->va...", G, self.values )
        return v if self.dim == 2 else np.einsum( "bj,vaj->vab", G, v )

    def refined( self, values, n ):
        """each pixel of `values [ nb_views, ... ]` split into `n` cells per axis, sharing its intensity"""
        for axis in range( 1, self.dim ):
            values = np.repeat( values, n, axis = axis ) / n
        return values

    def diracs( self, n = 1 ):
        """`( positions [ n nb_bins ], masses )`: the grid of the diracs along one detector axis ( the same for every axis
        and every view ) and their masses, of shape `[ nb_views, n nb_bins ( , n nb_bins ) ]`"""
        pos = self.origin + self.h / n * ( np.arange( n * self.nb_bins ) + 0.5 )
        return pos, self.refined( self.values, n )
