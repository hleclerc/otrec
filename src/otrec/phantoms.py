"""KNOWN 3D OBJECTS for the reconstruction experiments: their radiographs, a sample of them, and how far a cloud is from them.

A phantom is a set of uniform balls and cylinders ( axis `z`, the rotation axis ), each with a density that may be negative:
a hollow ball is a ball minus a smaller one, a FOAM a cylinder minus disjoint balls ( its pores ). Ready-made sets: `foam`
( the object of interest: a hollow material ), and simpler ones to check the machinery: `balls` ( a few random disjoint
balls ), `hollow`, `touching` ( two balls in contact ), `thin` ( a chain of small balls ).
"""
import numpy as np

#: how many points at a time the densities and distances are evaluated on ( [ chunk, nb_balls ] arrays )
_CHUNK = 16384


class Phantom:
    """`centers`, `radii`, `densities`: the balls; `cylinders`: rows `( x, y, z, radius, height, density )`.
    `detector`: the side of the detector its radiographs are taken on ( the object must fit in it )."""

    def __init__( self, centers, radii, densities = None, cylinders = None, name = "phantom", detector = 4.0 ):
        self.centers = np.asarray( centers, dtype = float ).reshape( -1, 3 )
        self.radii = np.asarray( radii, dtype = float ).reshape( -1 )
        self.densities = np.ones( len( self.radii ) ) if densities is None else np.asarray( densities, dtype = float )
        self.cylinders = np.zeros( ( 0, 6 ) ) if cylinders is None else np.asarray( cylinders, dtype = float ).reshape( -1, 6 )
        self.name = name
        self.detector = float( detector )

    def _boxes( self ):
        """the bounding boxes of the positive primitives, `[ m, 2, 3 ]`"""
        pos = self.densities > 0
        b = [ np.stack( [ self.centers[ pos ] - self.radii[ pos, None ], self.centers[ pos ] + self.radii[ pos, None ] ], 1 ) ]
        c = self.cylinders[ self.cylinders[ :, 5 ] > 0 ]
        if len( c ):
            half = np.stack( [ c[ :, 3 ], c[ :, 3 ], c[ :, 4 ] / 2 ], 1 )
            b.append( np.stack( [ c[ :, :3 ] - half, c[ :, :3 ] + half ], 1 ) )
        return np.concatenate( b )

    @property
    def extent( self ):
        """the side of a centered cube that holds the object, with a margin ( where to look for its visual hull )"""
        return float( 2.1 * np.abs( self._boxes() ).max() )

    # -- the data -------------------------------------------------------------------------------

    def radiographs( self, nb_angles = 60, nb_pixels = 256, extent = None, quadrature = 4 ):
        from .Radiographs import Radiographs
        r = Radiographs( nb_angles = nb_angles, nb_u = nb_pixels, nb_v = nb_pixels,
                         extent_u = self.detector if extent is None else extent, quadrature = quadrature )
        for c in self.cylinders:
            r.add_cylinder( c[ :3 ], c[ 3 ], c[ 4 ], density = c[ 5 ] )
        for c, rad, d in zip( self.centers, self.radii, self.densities ):
            r.add_sphere( c, rad, density = d )
        return r

    def _cylinder_dist( self, x ):
        """signed distances to the cylinders, `[ n, m ]` ( < 0 inside )"""
        c = self.cylinders
        dr = np.linalg.norm( x[ :, None, :2 ] - c[ None, :, :2 ], axis = 2 ) - c[ None, :, 3 ]
        dz = np.abs( x[ :, None, 2 ] - c[ None, :, 2 ] ) - c[ None, :, 4 ] / 2
        out = np.hypot( np.maximum( dr, 0 ), np.maximum( dz, 0 ) )
        return np.where( ( dr < 0 ) & ( dz < 0 ), np.maximum( dr, dz ), out )

    def _chunked( self, f, x ):
        x = np.asarray( x, dtype = float )
        return np.concatenate( [ f( x[ i:i + _CHUNK ] ) for i in range( 0, len( x ), _CHUNK ) ] ) if len( x ) else np.zeros( 0 )

    def density( self, x ):
        def f( x ):
            d2 = ( ( x[ :, None, : ] - self.centers[ None ] ) ** 2 ).sum( 2 )
            out = ( ( d2 <= self.radii[ None ] ** 2 ) * self.densities[ None ] ).sum( 1 )
            if len( self.cylinders ):
                out = out + ( ( self._cylinder_dist( x ) <= 0 ) * self.cylinders[ None, :, 5 ] ).sum( 1 )
            return out
        return self._chunked( f, x )

    def sample( self, n, seed = 0 ):
        """`n` points drawn from the density ( by rejection in the bounding box )"""
        rng = np.random.default_rng( seed )
        boxes = self._boxes()
        lo, hi = boxes[ :, 0 ].min( 0 ), boxes[ :, 1 ].max( 0 )
        top = float( self.densities.clip( 0 ).sum() + self.cylinders[ :, 5 ].clip( 0 ).sum() )   # a bound of the density
        out = [ ]
        while sum( len( o ) for o in out ) < n:
            x = rng.uniform( lo, hi, ( 4 * n + 1024, 3 ) )
            keep = rng.random( len( x ) ) * top < self.density( x )
            out.append( x[ keep ] )
        return np.concatenate( out )[ :n ]

    # -- how far a cloud is -----------------------------------------------------------------------

    def distance_to_support( self, x ):
        """for each point, how far it is from where the density is positive ( 0 inside ): out of the positive primitives,
        the distance to them; in a hole ( a negative ball, a pore ), the depth into it"""
        pos, neg = self.densities > 0, self.densities < 0

        def f( x ):
            d = np.linalg.norm( x[ :, None, : ] - self.centers[ None ], axis = 2 ) - self.radii[ None ]    # [ n, balls ]
            parts = [ d[ :, pos ] ]
            if len( self.cylinders ):
                parts.append( self._cylinder_dist( x ) )
            out = np.maximum( np.concatenate( parts, 1 ).min( 1 ), 0.0 )
            if neg.any():
                hole = np.maximum( -d[ :, neg ], 0.0 ).max( 1 )
                out = np.where( out == 0, hole, out )
            return out

        return self._chunked( f, x )

    def metrics( self, sample_size = 20000, seed = 1, tol = 0.0 ):
        """`truth( points ) -> dict`, for `Reconstruction3d.lloyd( truth = ... )`:

        - `inside`: the fraction of the points within `tol` of the object ( a pixel, say: the data does not see finer );
          `dist_mean`, `dist_max`: their distance to it;
        - `coverage`: the mean distance from a sample of the object to the nearest point ( a hole the cloud leaves );
        - `balance` ( balls only ): the largest gap, ball by ball ( the nearest positive center ), between the share of the
          points and the share of the object's mass.
        """
        from scipy.spatial import cKDTree
        ref = self.sample( sample_size, seed = seed )
        pos = np.flatnonzero( self.densities > 0 )
        with_balance = len( self.cylinders ) == 0 and len( pos ) > 1

        def shares( x ):
            near = cKDTree( self.centers[ pos ] ).query( x )[ 1 ]
            return np.bincount( near, minlength = len( pos ) ) / len( x )

        share = shares( ref ) if with_balance else None

        def truth( x ):
            x = np.asarray( x )
            d = self.distance_to_support( x )
            cov = cKDTree( x ).query( ref )[ 0 ]
            res = dict( inside = float( ( d <= tol ).mean() ), dist_mean = float( d.mean() ), dist_max = float( d.max() ),
                        coverage = float( cov.mean() ) )
            if with_balance:
                res[ "balance" ] = float( np.abs( shares( x ) - share ).max() )
            return res

        return truth


def foam( radius = 0.8, height = 1.4, pore_radii = ( 0.03, 0.15 ), wall = 0.015, porosity = 0.6, seed = 0, nb_rounds = 400,
          batch = 4096 ):
    """A FOAM: a cylinder of axis `z` with disjoint spherical pores, inside the detector's field ( side 2 ). The pores are
    placed by random sequential addition, at least `wall` apart and from the cylinder's surface: `nb_rounds` batches of
    candidates, their radius shrinking from round to round ( the large pores first, the small ones in the gaps ), until
    `porosity`."""
    rng = np.random.default_rng( seed )
    vol = np.pi * radius ** 2 * height
    C, R = np.zeros( ( 0, 3 ) ), np.zeros( 0 )
    filled = 0.0
    for t in range( nb_rounds ):
        hi = pore_radii[ 1 ] - ( pore_radii[ 1 ] - pore_radii[ 0 ] ) * t / max( 1, nb_rounds - 1 )
        r = rng.uniform( max( pore_radii[ 0 ], 0.8 * hi ), hi, batch )
        room = radius - r - wall
        rho, phi = room * np.sqrt( rng.random( batch ) ), 2 * np.pi * rng.random( batch )
        p = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ), rng.uniform( -1, 1, batch ) * ( height / 2 - r - wall ) ], 1 )
        if len( C ):   # against the pores already placed
            ok = ( np.linalg.norm( p[ :, None ] - C[ None ], axis = 2 ) - R[ None ] - r[ :, None ] ).min( 1 ) >= wall
            p, r = p[ ok ], r[ ok ]
        # then among themselves: greedily, in order, each accepted candidate blocking those it is too close to
        conflict = np.linalg.norm( p[ :, None ] - p[ None ], axis = 2 ) - r[ :, None ] - r[ None ] < wall
        keep, blocked = np.zeros( len( r ), bool ), np.zeros( len( r ), bool )
        for i in range( len( r ) ):
            if blocked[ i ]:
                continue
            keep[ i ] = True
            blocked |= conflict[ i ]
            filled += 4 / 3 * np.pi * r[ i ] ** 3
            if filled >= porosity * vol:
                break
        if len( r ):
            C, R = np.concatenate( [ C, p[ keep ] ] ), np.concatenate( [ R, r[ keep ] ] )
        if filled >= porosity * vol:
            break
    return Phantom( C, R, -np.ones( len( R ) ), cylinders = [ [ 0, 0, 0, radius, height, 1.0 ] ],
                    name = f"foam{ len( R ) }", detector = 2.0 )


def balls( nb = 3, seed = 0, rmin = 0.25, rmax = 0.5, box = 1.0 ):
    """`nb` random disjoint balls in `[ -box, box ]^3`"""
    rng = np.random.default_rng( seed )
    c, r = [ ], [ ]
    while len( c ) < nb:
        ci, ri = rng.uniform( -box + rmax, box - rmax, 3 ), rng.uniform( rmin, rmax )
        if all( np.linalg.norm( ci - cj ) > ri + rj + 0.05 for cj, rj in zip( c, r ) ):
            c.append( ci )
            r.append( ri )
    return Phantom( c, r, name = f"balls{ nb }" )


def shells( nb = 3, seed = 0, rmin = 0.25, rmax = 0.5, thickness = 0.05, box = 1.0 ):
    """`nb` random disjoint hollow balls in `[ -box, box ]^3`: shells of `thickness`, each a ball minus a ball of density -1"""
    rng = np.random.default_rng( seed )
    c, r = [ ], [ ]
    while len( c ) < nb:
        ci, ri = rng.uniform( -box + rmax, box - rmax, 3 ), rng.uniform( rmin, rmax )
        if all( np.linalg.norm( ci - cj ) > ri + rj + 0.05 for cj, rj in zip( c, r ) ):
            c.append( ci )
            r.append( ri )
    ph = Phantom( c + c, r + [ ri - thickness for ri in r ], [ 1.0 ] * nb + [ -1.0 ] * nb, name = f"shells{ nb }" )
    ph.thickness = thickness
    return ph


def hollow():
    """a hollow ball ( a shell ) and a full one"""
    return Phantom( [ [ 0.2, 0.0, 0.0 ], [ 0.2, 0.0, 0.0 ], [ -0.7, 0.3, 0.4 ] ], [ 0.7, 0.45, 0.25 ], [ 1.0, -1.0, 1.0 ], name = "hollow" )


def touching():
    """two balls in contact"""
    return Phantom( [ [ -0.35, 0.0, 0.1 ], [ 0.35, 0.0, -0.1 ] ], [ 0.35, 0.35 ], name = "touching" )


def thin( nb = 12 ):
    """a chain of small balls along a helix: a fine structure"""
    t = np.linspace( 0, 2 * np.pi, nb, endpoint = False )
    c = np.stack( [ 0.7 * np.cos( t ), 0.7 * np.sin( t ), 0.25 * ( t - np.pi ) ], axis = 1 )
    return Phantom( c, np.full( nb, 0.09 ), name = "thin" )


class VoronoiFoam( Phantom ):
    """A CLOSED-CELL FOAM: the faces of a Voronoi diagram, thickened to walls of thickness `thickness`, cut by a cylinder of
    axis `z`. A point is solid when it lies in the cylinder and within `thickness / 2` of a face; the distance to the
    nearest face is exact from the two nearest seeds, `( d2^2 - d1^2 ) / ( 2 |s2 - s1| )`.

    `nb_cells` seeds drawn uniformly in the cylinder grown by a cell ( the cells cut by its surface look like the inner
    ones ); `porosity`: the thickness is set by bisection on a Monte-Carlo estimate so that this fraction of the
    cylinder is void.

    The walls are far too many and too thin for an analytic projection: `radiographs` VOXELIZES the object ( partial
    occupancy, linear in the signed distance, `voxels` per side of the detector ) and integrates it along the rays
    ( jax, on the default device ).
    """

    def __init__( self, nb_cells = 400, porosity = 0.8, radius = 0.8, height = 1.4, seed = 0, nb_mc = 400000, detector = 2.0,
                  seed_height = None ):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng( seed )
        # the seeds are drawn for a cylinder of height `seed_height` ( `height` by default ), `nb_cells` cells in it, and the
        # object is its middle slab of height `height`: thinner slabs of the same foam, same walls
        sh = float( height if seed_height is None else seed_height )
        cell = ( np.pi * radius ** 2 * sh / nb_cells ) ** ( 1 / 3 )
        R, H = radius + cell, sh + 2 * cell
        nb = int( nb_cells * R ** 2 * H / ( radius ** 2 * sh ) )
        rho, phi = R * np.sqrt( rng.random( nb ) ), 2 * np.pi * rng.random( nb )
        self.seeds = np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ), rng.uniform( -H / 2, H / 2, nb ) ], 1 )
        self._tree = cKDTree( self.seeds )
        super().__init__( np.zeros( ( 0, 3 ) ), np.zeros( 0 ), cylinders = [ [ 0, 0, 0, radius, height, 1.0 ] ],
                          name = f"voronoi{ nb_cells }-{ int( round( 100 * porosity ) ) }", detector = detector )
        # the thickness that gives the porosity
        x = self._in_cylinder( rng, nb_mc )
        fd = self.face_distance( x )
        self.thickness = float( 2 * np.quantile( fd, 1 - porosity ) )
        self.porosity = float( porosity )
        self.cell = cell

    def _in_cylinder( self, rng, n ):
        c = self.cylinders[ 0 ]
        rho, phi = c[ 3 ] * np.sqrt( rng.random( n ) ), 2 * np.pi * rng.random( n )
        return np.stack( [ rho * np.cos( phi ), rho * np.sin( phi ), rng.uniform( -c[ 4 ] / 2, c[ 4 ] / 2, n ) ], 1 )

    def face_distance( self, x ):
        """the distance from each point to the nearest face of the Voronoi diagram"""
        def f( x ):
            d, i = self._tree.query( x, k = 2, workers = -1 )
            gap = np.linalg.norm( self.seeds[ i[ :, 1 ] ] - self.seeds[ i[ :, 0 ] ], axis = 1 )
            return ( d[ :, 1 ] ** 2 - d[ :, 0 ] ** 2 ) / ( 2 * gap )
        x = np.asarray( x, dtype = float )
        step = 1 << 20
        return np.concatenate( [ f( x[ i:i + step ] ) for i in range( 0, len( x ), step ) ] ) if len( x ) else np.zeros( 0 )

    def density( self, x ):
        x = np.asarray( x, dtype = float )
        inside = self._cylinder_dist( x )[ :, 0 ] <= 0
        out = np.zeros( len( x ) )
        if inside.any():
            out[ inside ] = self.face_distance( x[ inside ] ) <= self.thickness / 2
        return out

    def distance_to_support( self, x ):
        x = np.asarray( x, dtype = float )
        dc = self._cylinder_dist( x )[ :, 0 ]
        out = np.maximum( dc, 0.0 )
        inside = dc <= 0
        if inside.any():
            out[ inside ] = np.maximum( self.face_distance( x[ inside ] ) - self.thickness / 2, 0.0 )
        return out

    def sample( self, n, seed = 0 ):
        rng = np.random.default_rng( seed )
        out = [ ]
        while sum( len( o ) for o in out ) < n:
            x = self._in_cylinder( rng, 4 * n + 4096 )
            out.append( x[ self.face_distance( x ) <= self.thickness / 2 ] )
        return np.concatenate( out )[ :n ]

    def occupancy( self, nb, nz = None ):
        """the voxels: a square of side `detector` cut in `nb^2` in ( x, y ), `nz` layers of the same side in z ( centered;
        `nb` by default ), the solid fraction of each ( linear in the signed distance to the walls and to the cylinder ),
        `[ nz, nx, ny ]` float32, and the voxel side"""
        h = self.detector / nb
        nz = nb if nz is None else int( nz )
        g = -self.detector / 2 + h * ( np.arange( nb ) + 0.5 )
        gz = h * ( np.arange( nz ) - ( nz - 1 ) / 2 )
        c = self.cylinders[ 0 ]
        zs = np.flatnonzero( np.abs( gz - c[ 2 ] ) < c[ 4 ] / 2 + h )
        xy = np.flatnonzero( np.abs( g ) < c[ 3 ] + h )
        X, Y = np.meshgrid( g[ xy ], g[ xy ], indexing = "ij" )
        vol = np.zeros( ( nz, nb, nb ), np.float32 )
        for k in zs:
            p = np.stack( [ X.ravel(), Y.ravel(), np.full( X.size, gz[ k ] ) ], 1 )
            dc = self._cylinder_dist( p )[ :, 0 ]
            near = dc < h
            sd = np.full( len( p ), np.inf )
            sd[ near ] = np.maximum( self.face_distance( p[ near ] ) - self.thickness / 2, dc[ near ] )
            occ = np.clip( 0.5 - sd / h, 0.0, 1.0 ).reshape( X.shape )
            vol[ k, xy[ 0 ]:xy[ -1 ] + 1, xy[ 0 ]:xy[ -1 ] + 1 ] = occ
        return vol, h

    def detector_rows( self, nb_pixels ):
        """the detector's height in pixels: the object's, plus a few ( a thin object needs few rows )"""
        pix = self.detector / nb_pixels
        return min( nb_pixels, int( np.ceil( ( self.cylinders[ 0, 4 ] + 8 * pix ) / pix / 2 ) ) * 2 )

    def voxels( self, nb_pixels ):
        """the voxels the radiographs of `nb_pixels` are made from ( 2 per pixel ): `( vol [ nz, nx, ny ], h, origin )`,
        `origin` the center of voxel ( 0, 0, 0 ) as ( x, y, z )"""
        nb = 2 * nb_pixels
        vol, h = self.occupancy( nb, 2 * self.detector_rows( nb_pixels ) )
        return vol, h, np.array( [ -self.detector / 2 + h / 2, -self.detector / 2 + h / 2, -h * ( vol.shape[ 0 ] - 1 ) / 2 ] )

    def slice_projections( self, axes, s_min, ds, nb_bins, sub = 2 ):
        """the line integrals of the SLICE z = 0 ( a 2D object ): for each detector axis `axes[ k ]` ( `[ A, 2 ]`, unit ),
        the integral along the rays perpendicular to it, averaged over each bin `[ s_min + j ds, s_min + ( j + 1 ) ds )`
        ( `[ A, nb_bins ]` ). The slice is voxelized at `sub` voxels per bin, the rays sampled every voxel side with a
        bilinear interpolation, `sub` rays per bin."""
        import jax
        import jax.numpy as jnp
        nb = int( round( self.detector / ds ) ) * sub
        img, h = self.occupancy( nb, 1 )
        I = jnp.asarray( img[ 0 ] )
        half = self.detector / 2
        ss = s_min + ( ds / sub ) * ( jnp.arange( nb_bins * sub ) + 0.5 )     # the rays
        ts = -half + h * ( jnp.arange( nb ) + 0.5 )                            # along a ray

        @jax.jit
        def one( e ):
            # p = s e + t e_perp, e_perp = ( -e_y, e_x ); in voxel index units
            x = ( ss[ :, None ] * e[ 0 ] - ts[ None, : ] * e[ 1 ] + half ) / h - 0.5
            y = ( ss[ :, None ] * e[ 1 ] + ts[ None, : ] * e[ 0 ] + half ) / h - 0.5
            i0, j0 = jnp.floor( x ).astype( jnp.int32 ), jnp.floor( y ).astype( jnp.int32 )
            fx, fy = x - i0, y - j0

            def at( i, j ):
                ok = ( i >= 0 ) & ( i < nb ) & ( j >= 0 ) & ( j < nb )
                return jnp.where( ok, I[ jnp.clip( i, 0, nb - 1 ), jnp.clip( j, 0, nb - 1 ) ], 0.0 )

            line = ( at( i0, j0 ) * ( 1 - fx ) * ( 1 - fy ) + at( i0 + 1, j0 ) * fx * ( 1 - fy )
                     + at( i0, j0 + 1 ) * ( 1 - fx ) * fy + at( i0 + 1, j0 + 1 ) * fx * fy ).sum( 1 ) * h
            return line.reshape( nb_bins, sub ).mean( 1 )

        return np.stack( [ np.asarray( one( jnp.asarray( e, dtype = jnp.float32 ) ) ) for e in np.asarray( axes ) ] ).astype( float )

    def radiographs( self, nb_angles = 60, nb_pixels = 256, extent = None, quadrature = None, voxels = None ):
        """voxelized ( see the class docstring ): `voxels` per side ( 2 per pixel by default ), each detector pixel the mean of
        2 x 2 rays, each ray sampled every voxel side with a bilinear interpolation in ( x, y )"""
        import jax
        import jax.numpy as jnp
        from .Radiographs import Radiographs
        if extent is not None and extent != self.detector:
            raise ValueError( "VoronoiFoam: the detector is the voxelized cube, `extent` cannot differ" )
        nb_v = self.detector_rows( nb_pixels )
        pix = self.detector / nb_pixels
        r = Radiographs( nb_angles = nb_angles, nb_u = nb_pixels, nb_v = nb_v, extent_u = self.detector, extent_v = nb_v * pix )
        nb = int( voxels or 2 * nb_pixels )
        if nb % nb_pixels:
            raise ValueError( "voxels must be a multiple of nb_pixels" )
        sub = nb // nb_pixels                             # voxel layers ( and rays ) per pixel
        vol, h = self.occupancy( nb, nb_v * sub )
        V = jnp.asarray( vol )
        half = self.detector / 2
        us = -half + h * ( jnp.arange( nb ) + 0.5 )       # the rays in u: one per voxel side
        ts = us                                           # along the ray, every voxel side

        @jax.jit
        def view( th ):
            c, s = jnp.cos( th ), jnp.sin( th )
            # ( x, y ) = u ( -s, c ) + t ( c, s ), in voxel index units
            x = ( -s * us[ :, None ] + c * ts[ None, : ] + half ) / h - 0.5
            y = ( c * us[ :, None ] + s * ts[ None, : ] + half ) / h - 0.5
            i0, j0 = jnp.floor( x ).astype( jnp.int32 ), jnp.floor( y ).astype( jnp.int32 )
            fx, fy = x - i0, y - j0

            def at( i, j ):
                ok = ( i >= 0 ) & ( i < nb ) & ( j >= 0 ) & ( j < nb )
                return jnp.where( ok[ None ], V[ :, jnp.clip( i, 0, nb - 1 ), jnp.clip( j, 0, nb - 1 ) ], 0.0 )

            line = ( ( at( i0, j0 ) * ( 1 - fx ) * ( 1 - fy ) + at( i0 + 1, j0 ) * fx * ( 1 - fy )
                       + at( i0, j0 + 1 ) * ( 1 - fx ) * fy + at( i0 + 1, j0 + 1 ) * fx * fy ) ).sum( 2 ) * h   # [ nz, nu ]
            # a pixel: the mean of its sub x sub rays ( v = z: the voxel layers )
            img = line.reshape( nb_v, sub, nb_pixels, sub ).mean( ( 1, 3 ) )                                 # [ v, u ]
            return img.T                                                                                     # [ u, v ]

        r.values = np.stack( [ np.asarray( view( th ) ) for th in r.angles ] ).astype( float )
        self.voxel_volume = float( vol.sum() ) * h ** 3
        return r


PHANTOMS = dict( voronoi = VoronoiFoam, foam = foam, balls = balls, hollow = hollow, touching = touching, thin = thin, shells = shells )
