import functools

import torch

from ..shapes import deposit_nodes, fused, gaussian_matrix


class Model:
    """What the transport compares: on the grid of the diracs ( `n_per_pixel` cells per pixel and per axis: `N` cells of
    side `h` from `origin` ), the image of the shapes of radius `radius` and the masses of the diracs.
    Tensors are float64, on the device of loom ( "cpu", "cuda"; None: loom's default, the card where there is one ). `view_batch`: how many views a 2D detector deposits at once ( `[ V, n, window ]` )."""

    def __init__( self, sinogram, radius, k = 2, n_per_pixel = 1, view_batch = 8, device = None ):
        self.sinogram, self.radius, self.k = sinogram, float( radius ), int( k )
        self.n_per_pixel, self.view_batch = int( n_per_pixel ), int( view_batch )
        self.N, self.h, self.origin = self.n_per_pixel * sinogram.nb_bins, sinogram.h / self.n_per_pixel, sinogram.origin
        # one device for the tensors and the kernels: loom's ( `device`, when given, becomes its default )
        import loom
        if device is not None:
            loom.default_device = device
        self.device = torch.device( "cuda" if loom.resolved_device().name == "Cuda" else "cpu" )
        self.axes = self.tensor( sinogram.axes )

    @property
    def grid( self ):
        """what the deposits take after the positions ( `radius` may change from a stage to the next )"""
        return ( self.N, self.h, self.origin, self.radius, self.k )

    def tensor( self, x ):
        return torch.as_tensor( x, dtype = torch.float64, device = self.device )

    def batches( self ):
        """the views, `view_batch` at a time, as slices"""
        nv = self.sinogram.nb_views
        return [ slice( i, min( i + self.view_batch, nv ) ) for i in range( 0, nv, self.view_batch ) ]

    @functools.lru_cache( maxsize = 8 )
    def _blur( self, sigma ):
        nb = self.N if self.sinogram.dim == 2 else self.N + 1
        return self.tensor( gaussian_matrix( nb, sigma * self.n_per_pixel ) )

    def images( self, points, sigma, background, views = slice( None ) ):
        """the images of the shapes in `views`: the mass per cell `[ V, N ]` for a 1D detector, the density at the nodes
        `[ V, N + 1, N + 1 ]` for a 2D one; blurred by `sigma` ( pixels ) and lifted by `background` times their mean"""
        if self.sinogram.dim == 2:
            return self.finish( fused.deposit( points, self.axes[ views, 0 ], *self.grid ), sigma, background )
        uv = torch.einsum( "vad,nd->vna", self.axes[ views ], points )                     # the projected centres
        return self.finish( deposit_nodes( uv, *self.grid ), sigma, background )

    def finish( self, deposit, sigma, background ):
        """a deposit `[ V, ... ]`, blurred by `sigma` ( pixels ) and lifted by `background` times its mean"""
        G = self._blur( float( sigma ) )
        v = deposit @ G.T if self.sinogram.dim == 2 else G @ deposit @ G.T
        return v + background * v.mean( dim = tuple( range( 1, v.ndim ) ), keepdim = True )

    @functools.lru_cache( maxsize = 4 )
    def data( self, sigma, background ):
        """the masses of the diracs at the blur `sigma`, lifted: `[ nb_views, N ( , N ) ]`"""
        b = self.sinogram.refined( self.sinogram.blurred( sigma ), self.n_per_pixel )
        return self.tensor( b + background * b.mean( axis = tuple( range( 1, b.ndim ) ), keepdims = True ) )
