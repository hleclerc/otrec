"""A MEASURED parallel-beam scan of one slice, and the steps that make it the `Sinogram` a reconstruction takes. Each step
gives a new `Scan`: an example writes them one after the other, in the order it needs.

    values[ view, pixel ]    line integrals of the attenuation ( -log of the transmission ); pixel i centred at i
    angles[ view ]           radians, over 180 or 360 degrees
    cor                      where the rotation axis falls on the detector, in pixels

    Scan.from_counts( projections, flats, darks, angles )    the counts of a camera -> line integrals
    rotation_axis()          the axis, found by matching views half a turn apart
    centred( cor, radius )   the pixels centred on the axis, over a disk that every view sees whole
    minus_air( fraction )    the level of the air ( the ends of each view ) taken off each view
    linearized()             beam hardening: `p + a p^2`, `a` such that every view holds the same mass
    clipped()                what is negative ( noise, phase fringes ) set to zero
    sinogram( nb_bins, nb_views )    360 degrees folded onto 180, pixels and views averaged down: a `Sinogram`

What the loss normalizes needs no step: a drift of the beam that scales a whole view."""
import numpy as np

from .Sinogram import Sinogram


class Scan:
    def __init__( self, values, angles, cor = None ):
        self.values = np.asarray( values, dtype = float )
        self.angles = np.asarray( angles, dtype = float )
        self.cor = ( self.values.shape[ 1 ] - 1 ) / 2 if cor is None else float( cor )
        if self.values.ndim != 2 or self.angles.shape != self.values.shape[ :1 ]:
            raise ValueError( "Scan: values [ nb_views, nb_pixels ] and angles [ nb_views ]" )

    def _with( self, values, cor = None ):
        return Scan( values, self.angles, self.cor if cor is None else cor )

    @classmethod
    def from_counts( cls, projections, flats, darks, angles, cor = None ):
        """`-log( ( projection - dark ) / ( flat - dark ) )`, the flats and darks `[ m, nb_pixels ]` averaged"""
        dark = np.mean( darks, 0 )
        beam = np.mean( flats, 0 ) - dark
        return cls( -np.log( np.maximum( np.asarray( projections, dtype = float ) - dark, 1.0 ) / beam ), angles, cor )

    @property
    def nb_pixels( self ):
        return self.values.shape[ 1 ]

    @property
    def is_centred( self ):
        return self.cor == ( self.nb_pixels - 1 ) / 2

    # -- the steps ---------------------------------------------------------------------------------------------------------

    def rotation_axis( self ):
        """`cor`: the view the nearest to half a turn after the first one is the first one mirrored about the axis
        ( a cross-correlation, refined between pixels )"""
        a = self.values[ 0 ]
        b = self.values[ np.argmin( np.abs( np.mod( self.angles - self.angles[ 0 ], 2 * np.pi ) - np.pi ) ) ][ ::-1 ]
        N = self.nb_pixels
        c = np.fft.irfft( np.fft.rfft( a - a.mean(), 2 * N ) * np.conj( np.fft.rfft( b - b.mean(), 2 * N ) ), 2 * N )
        k = int( np.argmax( c ) )
        y0, y1, y2 = c[ k - 1 ], c[ k ], c[ ( k + 1 ) % ( 2 * N ) ]
        shift = ( k if k < N else k - 2 * N ) + 0.5 * ( y0 - y2 ) / ( y0 - 2 * y1 + y2 )      # a( x ) = b( x - shift )
        return ( N - 1 + shift ) / 2

    def centred( self, cor = None, radius = None ):
        """the views resampled on pixels of the same width, centred on the axis `cor` ( default: `self.cor` ), over the
        disk of `radius` pixels ( default: the largest that every view sees whole; less when the object is smaller )"""
        cor = self.cor if cor is None else float( cor )
        radius = np.floor( min( cor + 0.5, self.nb_pixels - 0.5 - cor, np.inf if radius is None else radius ) )
        cum = np.concatenate( [ np.zeros( ( len( self.values ), 1 ) ), np.cumsum( self.values, 1 ) ], 1 )
        old, new = np.arange( self.nb_pixels + 1 ) - 0.5 - cor, np.arange( -radius, radius + 1 )
        values = np.diff( np.stack( [ np.interp( new, old, c ) for c in cum ] ), axis = 1 )
        return self._with( values, cor = radius - 0.5 )

    def minus_air( self, fraction = 0.03 ):
        """the median of the `fraction` of the pixels at each end of a view, where there is only air, taken off the view"""
        k = max( 1, int( fraction * self.nb_pixels ) )
        air = np.median( np.concatenate( [ self.values[ :, :k ], self.values[ :, -k: ] ], 1 ), 1, keepdims = True )
        return self._with( self.values - air )

    def linearized( self ):
        """beam hardening ( a polychromatic beam: the thicker the matter, the less it attenuates per unit of length ):
        `p + a p^2`, with `a` making the mass of the views as equal as it can"""
        s1, s2 = self.values.sum( 1 ), ( self.values ** 2 ).sum( 1 )
        a = -np.cov( s1, s2 )[ 0, 1 ] / np.var( s2, ddof = 1 )
        return self._with( self.values + a * self.values ** 2 )

    def clipped( self ):
        return self._with( np.maximum( self.values, 0.0 ) )

    def sinogram( self, nb_bins = 512, nb_views = None ):
        """the `Sinogram` of a centred scan, of extent 2 ( the disk of radius 1 ): each view at `t + pi` is the view at
        `t` mirrored; the views grouped by angle into `nb_views` ( default: as many as over half a turn ) and averaged;
        the pixels summed into `nb_bins`"""
        if not self.is_centred:
            raise ValueError( "Scan.sinogram: centre it first ( `centred` )" )
        values = self.values.copy()
        u = np.mod( self.angles / np.pi + 1e-9, 2.0 )                   # in half turns ( a view on a limit: the next one )
        behind = u >= 1
        values[ behind ] = values[ behind, ::-1 ]
        u -= behind

        nb_views = int( nb_views or ( ~behind ).sum() )
        group = np.minimum( np.floor( u * nb_views + 1e-6 ).astype( int ), nb_views - 1 )
        count = np.bincount( group, minlength = nb_views )
        if ( count == 0 ).any():
            raise ValueError( f"Scan.sinogram: { ( count == 0 ).sum() } of the { nb_views } groups of angles are empty" )
        views = np.zeros( ( nb_views, self.nb_pixels ) )
        np.add.at( views, group, values )
        views /= count[ :, None ]
        angle = np.pi * np.bincount( group, u, nb_views ) / count

        # the pixels into bins: the integral read where the edges of the bins fall; the mass of a bin is the line integral
        # times its width, in the frame where the disk has radius 1
        N = self.nb_pixels
        cum = np.concatenate( [ np.zeros( ( nb_views, 1 ) ), np.cumsum( views, 1 ) ], 1 )
        edges = np.linspace( 0, N, nb_bins + 1 )
        bins = np.diff( np.stack( [ np.interp( edges, np.arange( N + 1 ), c ) for c in cum ] ), axis = 1 ) * ( 2 / N )
        axes = np.stack( [ np.cos( angle ), np.sin( angle ) ], -1 )[ :, None, : ]
        return Sinogram( 2, nb_views, nb_bins, 2.0, axes = axes, values = bins )
