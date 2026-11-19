"""A MEASURED parallel-beam scan of one slice, and the steps that make it the `Sinogram` a reconstruction takes. Each step
gives a new `Scan`: an example writes them one after the other, in the order it needs.

    values[ view, pixel ]    line integrals of the attenuation ( -log of the transmission ); pixel i centred at i
    angles[ view ]           radians, over 180 or 360 degrees
    cor                      where the rotation axis falls on the detector, in pixels

    Scan.from_counts( projections, flats, darks, angles )    the counts of a camera -> line integrals
    rotation_axis()          the axis, found by matching views half a turn apart
    aligned()                the views shifted so that their centres of mass follow the sinusoid of a still object
                             ( matter entirely in the field ): the axis, and the motion of the sample along the detector
    rings_removed( width )   the bias of each pixel of the detector ( what makes rings ) taken off every view
    centred( cor, radius )   the pixels centred on the axis, over a disk that every view sees whole
    minus_air( fraction )    the level of the air ( the ends of each view ) taken off each view
    linearized()             beam hardening: `p + a p^2`, `a` such that every view holds the same mass
    clipped()                what is negative ( noise, phase fringes ) set to zero
    phase_retrieved( a )     phase contrast ( fringes on the edges ) -> a density: Paganin's filter along the detector
    outside_removed()        matter bigger than the field: what the matter outside the disk adds, estimated and taken off
    moments_matched()        then the same mass in every view, its centre on a sinusoid ( Helgason-Ludwig, orders 0, 1 )
    zeroed_air( k )          what is under `k` times the noise of the air set to zero: the air at 0, not at the mean of
                             its clipped noise ( a thin layer of matter all over the disk, which draws shapes into it )
    equal_masses()           each view scaled to the mean mass: what an object entirely in the field gives
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

    def aligned( self, threshold = 0.0 ):
        """a still object in parallel beam: the centre of mass of view `t` is `cor + a cos t + b sin t` ( its mass the
        same in every view ). The sinusoid is fitted on the views; each view is shifted onto it, and `cor` is its centre.
        What it corrects: an axis given wrong, and the motion of the sample along the detector ( drift, vibrations ). The
        matter must be entirely in the field, the air at 0 ( `minus_air`, `zeroed_air` ); the values under `threshold`
        are left out of the centres"""
        x = np.arange( self.nb_pixels, dtype = float )
        w = np.where( self.values > threshold, self.values, 0.0 )
        c = ( w * x ).sum( 1 ) / np.maximum( w.sum( 1 ), 1e-300 )
        A = np.stack( [ np.ones_like( self.angles ), np.cos( self.angles ), np.sin( self.angles ) ], 1 )
        coef = np.linalg.lstsq( A, c, rcond = None )[ 0 ]
        shift = A @ coef - c                                                      # where each view must go
        cum = np.concatenate( [ np.zeros( ( len( self.values ), 1 ) ), np.cumsum( self.values, 1 ) ], 1 )
        edges = np.arange( self.nb_pixels + 1 ) - 0.5
        values = np.diff( np.stack( [ np.interp( edges - d, edges, cu ) for cu, d in zip( cum, shift ) ] ), axis = 1 )
        out = self._with( values, cor = coef[ 0 ] )
        out.shifts = shift
        return out

    def rings_removed( self, width = 9 ):
        """the RINGS: a pixel of the detector that reads a bit too much or too little does it in every view ( a stripe of
        the sinogram, a ring in the image ). Its bias is what the mean view departs from a smooth curve ( a polynomial
        of degree 3 over `width` pixels, Savitzky-Golay: the curvature of the object stays ); it is taken off every
        view. ( A bias that varies slowly along the detector stays, as would a detail of the object as thin as a pixel
        sitting on the axis, the same in every view )"""
        from scipy.signal import savgol_filter
        mean = self.values.mean( 0 )
        return self._with( self.values - ( mean - savgol_filter( mean, width, 3, mode = "interp" ) )[ None ] )

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

    def zeroed_air( self, k = 3.0, fraction = 0.03 ):
        """the values under `k sigma` set to zero, `sigma` the noise of the air ( a robust deviation of the `fraction` of
        the pixels at each end of the views, all views together: take the air off first, `minus_air` ). Negative values
        go too ( a clip ). The matter loses the little of it that is under the threshold"""
        n = max( 1, int( fraction * self.nb_pixels ) )
        air = np.concatenate( [ self.values[ :, :n ], self.values[ :, -n: ] ], 1 )
        sigma = 1.4826 * np.median( np.abs( air - np.median( air ) ) )
        return self._with( np.where( self.values < k * sigma, 0.0, self.values ) )

    def equal_masses( self ):
        """each view scaled so that its mass is the mean one ( the matter entirely in the field: every view sees all of
        it ). ( The loss normalizes each view: this changes what the FBP and the pages see, not the reconstruction )"""
        m = self.values.sum( 1, keepdims = True )
        return self._with( self.values * ( m.mean() / np.where( m > 0, m, 1.0 ) ) )

    def phase_retrieved( self, a ):
        """PHASE CONTRAST ( a beam that travels after the sample: each edge drawn by a bright and a dark fringe ) turned into
        a density: each view filtered by `1 / ( 1 + a k^2 )` along the detector ( Paganin, `a` in pixels^2: `delta / beta`
        times the distance times the wavelength over 4 pi ). In 1D it is Paganin's filter exactly where the object does not
        vary along the axis ( fibres along it ); elsewhere the fringes are only lessened"""
        N = self.nb_pixels
        k = 2 * np.pi * np.fft.rfftfreq( 2 * N )
        mirrored = np.concatenate( [ self.values, self.values[ :, ::-1 ] ], 1 )             # ( no jump at the ends )
        return self._with( np.fft.irfft( np.fft.rfft( mirrored, axis = 1 ) / ( 1 + a * k * k ), 2 * N, axis = 1 )[ :, :N ] )

    def outside_removed( self, pad = 1.0, nb_bins = 512, nb_views = 360, edge = 8 ):
        """matter BIGGER THAN THE FIELD: what the matter outside the disk adds to the views, taken off them, so that they are
        the projections of the disk alone ( what the shapes can give ). The views, `nb_bins` and `nb_views` of them, are
        extended by `pad` radii on each side ( the value at their ends, mean of `edge` bins, falling to 0 as a cosine ); the
        filtered back projection of that, outside the disk, projected back, is the estimate, interpolated onto the views.
        It is wrong where the extension is ( the shape of the sample is not known ): by a smooth amount, which
        `moments_matched` corrects"""
        from .projection import filtered_backprojection, project_image
        if not self.is_centred:
            raise ValueError( "Scan.outside_removed: centre it first ( `centred` )" )
        sino = self.sinogram( nb_bins = nb_bins, nb_views = nb_views )
        W = int( round( pad * nb_bins / 2 ) )
        taper = np.cos( 0.5 * np.pi * ( np.arange( W ) + 0.5 ) / W ) ** 2
        v = sino.values
        ends = v[ :, :edge ].mean( 1, keepdims = True ) * taper[ ::-1 ], v[ :, -edge: ].mean( 1, keepdims = True ) * taper
        n, E, directions = nb_bins + 2 * W, 2.0 * ( nb_bins + 2 * W ) / nb_bins, sino.axes[ :, 0, : ]
        image = filtered_backprojection( np.concatenate( [ ends[ 0 ], v, ends[ 1 ] ], 1 ), E, directions, n )
        x = -E / 2 + E / n * ( np.arange( n ) + 0.5 )
        image[ x[ :, None ] ** 2 + x[ None, : ] ** 2 <= 1.0 ] = 0.0
        outside = project_image( image, E, directions, n )[ :, W:W + nb_bins ]

        # onto the views of the scan: a view at t + pi is the one at t mirrored; linear in the angle ( periodic over
        # half a turn, with the mirror ), and in the position along the detector. ( Masses of bins -> line integrals )
        outside = outside / ( 2 / nb_bins )
        t = np.mod( np.arctan2( directions[ :, 1 ], directions[ :, 0 ] ), np.pi )
        order = np.argsort( t )
        t, outside = np.concatenate( [ t[ order ], [ t[ order[ 0 ] ] + np.pi ] ] ), np.concatenate( [ outside[ order ], outside[ order[ :1 ], ::-1 ] ] )
        t, outside = np.concatenate( [ [ t[ -2 ] - np.pi ], t ] ), np.concatenate( [ outside[ -2: -1, ::-1 ], outside ] )
        bins = -1 + 2 / nb_bins * ( np.arange( nb_bins ) + 0.5 )
        pixels = -1 + 2 / self.nb_pixels * ( np.arange( self.nb_pixels ) + 0.5 )
        values = self.values.copy()
        for i, angle in enumerate( self.angles ):
            u = np.mod( angle, 2 * np.pi )
            behind = u >= np.pi
            u -= np.pi * behind
            j = min( max( int( np.searchsorted( t, u ) ), 1 ), len( t ) - 1 )
            f = ( u - t[ j - 1 ] ) / ( t[ j ] - t[ j - 1 ] )
            row = ( 1 - f ) * outside[ j - 1 ] + f * outside[ j ]
            values[ i ] -= np.interp( -pixels if behind else pixels, bins, row )
        return self._with( values )

    def moments_matched( self ):
        """the first two conditions of Helgason and Ludwig: the projections of the disk have the same MASS in every view, and
        their CENTRE of mass on a sinusoid. What departs from them ( what is left of `outside_removed` ) is taken off each
        view as a smooth profile that vanishes at the ends, as the projections of the disk do: `1 - s^2` for the mass,
        `s ( 1 - s^2 )` for the centre ( `s` along the detector, 1 at the ends ). ( The chords of an annulus around the disk,
        closer to where the error comes from, are not 0 at the ends: an edge in the views, a ring in the image ). The
        level stays: the mean mass of the views"""
        if not self.is_centred:
            raise ValueError( "Scan.moments_matched: centre it first ( `centred` )" )
        s = -1 + 2 / self.nb_pixels * ( np.arange( self.nb_pixels ) + 0.5 )
        w = 1 - s * s
        m, m1 = self.values.sum( 1 ), ( self.values * s ).sum( 1 )
        A = np.stack( [ np.ones_like( self.angles ), np.cos( self.angles ), np.sin( self.angles ) ], 1 )
        centre = A @ np.linalg.lstsq( A, m1 / m, rcond = None )[ 0 ]
        even = ( m - m.mean() ) / w.sum()
        odd = ( m1 - m.mean() * centre ) / ( s * s * w ).sum()
        return self._with( self.values - even[ :, None ] * w - odd[ :, None ] * ( s * w ) )

    def sinogram( self, nb_bins = None, nb_views = None, reduction = 1 ):
        """the `Sinogram` of a centred scan, of extent 2 ( the disk of radius 1 ): each view at `t + pi` is the view at
        `t` mirrored; the views grouped by angle into `nb_views` and averaged; the pixels summed into `nb_bins`. By
        default, the data as they are: a bin per pixel, a view per view over half a turn -- or `reduction` times fewer
        of both ( a faster run, to try things )"""
        if not self.is_centred:
            raise ValueError( "Scan.sinogram: centre it first ( `centred` )" )
        values = self.values.copy()
        u = np.mod( self.angles / np.pi + 1e-9, 2.0 )                   # in half turns ( a view on a limit: the next one )
        behind = u >= 1
        values[ behind ] = values[ behind, ::-1 ]
        u -= behind

        nb_views = int( nb_views or max( 1, ( ~behind ).sum() // reduction ) )
        nb_bins = int( nb_bins or max( 1, self.nb_pixels // reduction ) )
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
