"""Tomographic reconstruction of a cloud of shapes from a `Sinogram`, in 2D or 3D, by minimizing

    L( shapes ) = sum_views  W2^2(  diracs of the view ,  image of the projected shapes  )            ( see `loss` )

A shape is a small piece of matter: the object ( continuous, with holes ) is the sum of 1e5 to 1e7 of them.
The constructor takes the PROBLEM ( the data, the profile of the shapes, the device ) and a `Recorder`, which keeps the
history; HOW it is solved is the sequence of the steps, each one from the current cloud ( `points [ n, dim ]`,
`radii [ n ]` ), chainable:

    rec = Reconstruction( sino )
    rec.add_random_balls( 1250, radius = 0.05 )                               # a random start, in what every view sees
    rec.adjust_parameters( centers = True, x_tol = 0.1 )                      # a STAGE: L-BFGS until nothing moves
    rec.split_items( division = 2, noise_ratio = 1.0 )                        # each ball in two, the radius / sqrt( 2 )
    rec.remove_isolated()                                                     # the balls alone in the air
    rec.adjust_parameters( centers = True, x_tol = 0.1 )
    rec.adjust_parameters( centers = True, cov_matrices = True, x_tol = 0.01 )  # ellipses: a covariance each ( 2D )

`adjust_parameters` says what moves: the centres, a radius each ( `radii`, 2D ), a covariance each ( `cov_matrices`,
2D: ellipses with the profile of the balls ), a density each ( `densities`, 2D, with either: an object that is not
binary ). `fit_polygons` makes star polygons of density 1 ( 2D ). `multiscale` is a
recipe: a random start, then stages and splits until a number of balls, each stage at the radius the cloud finds
( `nominal_radius` ).

`x_tol`: a stage stops when 99.9 % of the shapes have moved by less than that fraction of the RADIUS over 10
iterations ( or after `max_iter` ). `sigma` is a blur applied to both measures, for a stage that starts far from the
object. Every length is in WORLD units ( `radius`, `sigma` ) or relative to the shapes ( `x_tol`, `noise_ratio` ):
nothing is counted in pixels of the sinogram, whose number of bins is a parameter like another. Whatever the stage, the
shapes stay inside what every view sees ( `shapes.balls.inside` ).
The cloud is numpy arrays between the stages; a stage runs on tensors, on the device of loom ( see `loss.Model` )."""
import math
import time

import numpy as np
import torch

from . import cloud
from .lbfgs import lbfgs
from .loss import Loss1d, Model, make_loss
from .Recorder import Recorder
from .shapes import balls, ellipses, polygons, sized


class Reconstruction:
    """`points`, `radius`: a cloud to start from ( or `add_random_balls` ). Polygons and sized balls stay within
    `max_ratio` of the radius. `background`: both measures are lifted by that fraction of their mean. `recorder`: keeps the
    history ( `Recorder()` by default: the steps and the losses, no frames ). Random draws take `rng` ( seeded with 0 ).
    `outer` ( world 2D ): the radius where the shapes may be, beyond the field ( 1 for the sinogram of a `Scan` ), for an
    object bigger than it: only the window of the detector is compared ( `loss.Model` ). `fused`: the stages of balls of one
    radius evaluate the loss in one kernel ( `loss.LossFused`: the deposit, the transport and the gather together )"""

    def __init__( self, sinogram, points = None, *, radius = None, k = 2, n_per_pixel = 1, background = 1e-3, view_batch = 8,
                  device = None, max_ratio = 4.0, outer = None, recorder = None, fused = False ):
        self.sinogram, self.dim = sinogram, sinogram.dim
        self.model = Model( sinogram, radius, k, n_per_pixel, view_batch, device, max_ratio, outer )
        self.loss_fn, self.fused = make_loss( self.model, fused ), bool( fused )
        self.background = float( background )
        self.recorder = Recorder() if recorder is None else recorder
        self.recorder.dim, self.recorder.floor = self.dim, self.loss_fn.floor
        self.rng = np.random.default_rng( 0 )
        self.points = self.radii = self.ellipses = self.densities = None
        if points is not None:
            self._set( np.array( points, dtype = float ) )

    # -- the cloud ---------------------------------------------------------------------------------------------------------

    @property
    def radius( self ):
        """of the balls; set, every shape becomes a ball of that radius"""
        return self.model.radius

    @radius.setter
    def radius( self, value ):
        self.model.radius = float( value )
        if self.points is not None:
            self._set( self._inside( self.points ) )

    def _inside( self, points ):
        """the centres brought back where a ball of `self.radius` stays inside what every view sees. A stage must start
        there: beyond, the projection of the shapes ( `balls.inside` ) leaves a centre no radial derivative, and it
        would never come back ( balls drawn at radius 0, then given one )"""
        room = max( self.model.limit - self.radius, 0.0 ) / np.maximum( np.linalg.norm( points, axis = 1 ), 1e-300 )
        return points * np.minimum( room, 1.0 )[ :, None ]

    @property
    def nb_points( self ):
        return 0 if self.points is None else self.points.shape[ 0 ]

    def _set( self, points, radii = None, densities = None ):
        """the cloud: balls ( of `self.radius`, or of `radii` ), of density 1 ( or `densities` ); the ellipses are forgotten"""
        self.points, self.radii, self.ellipses = points, np.full( len( points ), self.radius ) if radii is None else radii, None
        self.densities = densities
        return self

    def add_random_balls( self, n, radius ):
        """`n` balls drawn uniformly in what every view sees ( the centred ball of diameter `extent`, less `radius` ), added
        to the cloud; every ball then has the radius `radius`"""
        new = cloud.uniform_in_ball( self.rng, n, self.dim, max( self.model.limit - radius, 0.0 ) )
        self.model.radius = float( radius )
        self._set( new if self.points is None else np.concatenate( [ self.points, new ] ) )
        self.recorder.step( "add_random_balls", n = n, nb_points = self.nb_points, radius = self.radius )
        return self

    def split_items( self, division = 2, noise_ratio = 1.0, n = None ):
        """each ball becomes `division` balls, moved by a gaussian of deviation `noise_ratio / 2` times the radius; with
        `n`, only `n` of them are kept. The radius is divided by `( new number / old number )^( 1 / dim )`: the mass of a
        ball is shared between those it becomes, at the same density"""
        nb = self.nb_points
        new = cloud.split( self.rng, self.points, division, noise_ratio * self.radius, n )
        self.model.radius = self.radius * ( nb / len( new ) ) ** ( 1 / self.dim )
        self._set( self._inside( new ) )
        self.recorder.step( "split_items", division = division, noise_ratio = noise_ratio, nb_points = self.nb_points, radius = self.radius )
        return self

    def remove_isolated( self, factor = 1.5 ):
        """the balls ALONE in the air taken off: those with no other ball within `factor` times the sum of their two radii
        ( 1.5: no neighbour within three radii, when the radii are the same ). A ball left there by a stage ( stuck
        between views that disagree ) holds a share of the mass that the matter is missing; the masses of the others
        grow to make up for it ( 1 in all ). ( Measured against the radii, not against the spacing of the cloud: the
        spacing of a dense part would make the balls of a lighter one look alone )"""
        from scipy.spatial import cKDTree
        d, j = cKDTree( self.points ).query( self.points, 2 )
        return self._remove( "remove_isolated", d[ :, 1 ] <= factor * ( self.radii + self.radii[ j[ :, 1 ] ] ), factor = factor )

    def remove_in_air( self, fraction = 0.0 ):
        """the balls OUTSIDE the matter taken off: those that more than `fraction` of the views see in empty air ( the
        field `seen empty`, `loss.fields` ). A ball outside the convex hull of the matter has such views; it needs air
        at 0 in the data ( `Scan.zeroed_air` ). The masses of the others grow to make up for it ( 1 in all )"""
        return self._remove( "remove_in_air", self.fields()[ "seen empty" ] <= fraction, fraction = fraction )

    def _remove( self, step, keep, **info ):
        """only the shapes `keep`, as they are ( radii, ellipses, densities ); the step, its loss, and a frame"""
        ell = self.ellipses
        self._set( self.points[ keep ], self.radii[ keep ], None if self.densities is None else self.densities[ keep ] )
        self.ellipses = None if ell is None else ell[ keep ]
        loss_fn, x = self._rows()
        loss = loss_fn.value_and_grad( x, 0.0, self.background )[ 0 ]
        self.recorder.step( step, snapshot = lambda: self._snapshot( loss_fn, x, 0.0, self.background ), removed = int( ( ~keep ).sum() ),
                            nb_points = self.nb_points, radius = self.radius, loss = loss, **info )
        return self

    def _rows( self ):
        """`( loss_fn, x )`: the cloud as the rows of its kind ( ellipses, sized balls, or balls of one radius )"""
        md = self.model
        if self.ellipses is not None:
            x = md.tensor( self.ellipses )
            if self.densities is not None:
                x = torch.cat( [ x, sized.DENSITY * self.radius * torch.log( md.tensor( self.densities ) )[ :, None ] ], 1 )
            return Loss1d( md, ellipses ), x
        if self.densities is not None or not np.all( self.radii == self.radius ):
            return Loss1d( md, sized ), self._sized()
        return self.loss_fn, md.tensor( self.points )

    def _snapshot( self, loss_fn, x, sigma, background ):
        """what a frame keeps: `( points, radii, drawing, fields )` ( `sigma` in pixels )"""
        md, kind = self.model, getattr( loss_fn, "kind", balls )
        with torch.no_grad():
            points, radii = ( t.cpu().numpy() for t in kind.discs( x, md ) )
        fields = loss_fn.fields( x, sigma, background ) if self.recorder.fields and hasattr( loss_fn, "fields" ) else { }
        return points, radii, kind.drawing( x, md ) if hasattr( kind, "drawing" ) else { }, fields

    def nominal_radius( self, area = None ):
        """the radius at which the balls FILL the matter: each one stands for a room, and a ball of that mass and of the
        density of the matter at its centre has that radius. The room: `area / nb_points` when the area ( volume ) of the
        matter is given ( `Sinogram.matter_area`, a binary object ), else from the distance between neighbours
        ( `cloud.first_peak`, `PEAK_ROOM` ). The cloud must be spread as the matter: balls smaller than the details of the
        object do that ( they place themselves as diracs would ); balls bigger than a strut line up along it, closer than
        the room they stand for, and stack. ( 2D: checked on packings and on a foam; 3D: the same formula, not calibrated. )
        With shapes around the field ( `outer` ), the distances are read on the balls of the field only: those around it,
        seen by some views only, are smeared along the circles and give a radius 20 to 35 % too large"""
        if area is not None:
            return ( area / self.nb_points / self.model.unit ) ** ( 1 / self.dim )
        points = self.points
        if self.model.pad:
            points = points[ np.linalg.norm( points, axis = 1 ) < 0.9 * self.sinogram.extent / 2 ]
        return ( cloud.first_peak( points ) ** self.dim / cloud.PEAK_ROOM / self.model.unit ) ** ( 1 / self.dim )

    # -- the loss ----------------------------------------------------------------------------------------------------------

    def _pixels( self, sigma ):
        """a blur in the unit of the loss: pixels of the sinogram"""
        return float( sigma ) / self.sinogram.h

    def value_and_grad( self, points = None, sigma = 0.0, background = None ):
        """the loss of balls of radius `self.radius` at `points [ n, dim ]` ( the centres of the cloud by default ), and its
        derivative"""
        p = self.model.tensor( self.points if points is None else points )
        f, g = self.loss_fn.value_and_grad( p, self._pixels( sigma ), self.background if background is None else float( background ) )
        return f, g.cpu().numpy()

    def image( self, sigma = 0.0, background = None ):
        """the image of the projected shapes ( the cloud, with its radii ) in every view, as the transport sees it"""
        md, bg = self.model, self.background if background is None else float( background )
        with torch.no_grad():
            if self.dim == 2:
                return md.finish( sized.deposit( *sized.shapes( self._sized(), md ), md ), self._pixels( sigma ), bg ).cpu().numpy()
            centres = balls.shapes( md.tensor( self.points ), md )[ 0 ]
            return torch.cat( [ md.images( centres, self._pixels( sigma ), bg, v ) for v in md.batches() ] ).cpu().numpy()

    def fields( self, sigma = 0.0, background = None ):
        """`{ name: [ n ] }`: what the views ask of each ball of the cloud ( `loss.fields`: move, dissonance, density;
        world 2D )"""
        bg = self.background if background is None else float( background )
        if self.dim != 2:
            return { }
        return Loss1d( self.model, sized ).fields( self._sized(), self._pixels( sigma ), bg )

    @property
    def loss( self ):
        """of the last stage"""
        stages = self.recorder.stages
        return stages[ -1 ][ "loss" ] if stages else None

    # -- the steps ---------------------------------------------------------------------------------------------------------

    def _sized_loss( self, grads ):
        """the loss of sized balls: fused when asked ( `grads`: the radii or densities move )"""
        if self.fused and self.dim == 2:
            from .loss.LossFused import LossFusedSized
            return LossFusedSized( self.model, grads )
        return Loss1d( self.model, sized )

    def _sized( self, densities = False ):
        """the rows of sized balls; with a density column when the cloud has densities, or when `densities`"""
        T = self.model.tensor
        d = self.densities if self.densities is not None else ( np.ones( self.nb_points ) if densities else None )
        return sized.pack( T( self.points ), T( self.radii ), self.radius, None if d is None else T( d ) )

    def _stage( self, step, info, loss_fn, x, sigma, background, x_tol, max_iter, fixed = ( ), callback = None, f_tol = None ):
        """L-BFGS on `x`, the rows of a kind of shapes ( `loss_fn.kind` ), its columns `fixed` kept ( or, a boolean mask
        `[ n, columns ]`, those of its entries ); sets the cloud from the result ( with the densities when the rows have
        them ), and returns it"""
        fixed = fixed if torch.is_tensor( fixed ) else list( fixed )
        t0, rc, md = time.perf_counter(), self.recorder, self.model
        bg = self.background if background is None else float( background )
        def fun( x ):
            f, g = loss_fn.value_and_grad( x, self._pixels( sigma ), bg )
            if torch.is_tensor( fixed ):
                g[ fixed ] = 0
            elif fixed:
                g[ :, fixed ] = 0
            return f, g
        kind = getattr( loss_fn, "kind", balls )
        def cloud_of( x ):                                             # `( points, radii )`
            with torch.no_grad():
                return tuple( t.cpu().numpy() for t in kind.discs( x, md ) )
        snapshot = lambda x: self._snapshot( loss_fn, x, self._pixels( sigma ), bg )
        def each( x, f ):
            rc.iteration_done( lambda: snapshot( x ), f )
            if callback:
                callback( x, f )
        loss0 = fun( x )[ 0 ]
        rc.begin_stage( step, dict( info, sigma = float( sigma ), x_tol = x_tol, nb_points = self.nb_points, radius = self.radius ), lambda: snapshot( x ), loss0 )
        project = ( lambda x: kind.project( x, md ) ) if hasattr( kind, "project" ) else None
        x, loss, nb_iter, nb_eval, stop = lbfgs( fun, x, x_tol * self.radius, max_iter, callback = each, project = project,
                                                 f_tol = f_tol, f_floor = self.loss_fn.floor if sigma == 0 else 0.0 )
        with torch.no_grad():
            dens = kind.weights( x, md ).cpu().numpy() if hasattr( kind, "weights" ) else None
        self._set( *cloud_of( x ), None if dens is None or ( dens == 1 ).all() else dens )
        rc.end_stage( lambda: snapshot( x ), loss0 = loss0, loss = loss, nb_iter = nb_iter, nb_eval = nb_eval, stop = stop, seconds = time.perf_counter() - t0 )
        return x

    def adjust_parameters( self, centers = False, radii = False, cov_matrices = False, densities = False, x_tol = 0.1, max_iter = 2000, sigma = 0.0,
                           background = None, callback = None, f_tol = None ):
        """a STAGE: L-BFGS on what is asked ( nothing moves unless it is said ), from the current cloud, at the blur `sigma`:

            centers         the centres
            radii           a radius each ( world 2D, `shapes.sized`; the mass follows the radius ). "keep": the radii of
                            the cloud as they are ( one each, fixed: balls of several sizes whose centres move; balls of
                            one radius when they all have `self.radius` )
            cov_matrices    a covariance each ( world 2D, `shapes.ellipses`: the profile of the balls, stretched; the mass
                            follows the area ). Round at the start, or the ellipses of the last stage. The result is
                            `self.ellipses [ n, 5 ]`; `points` and `radii` then describe the discs of the same areas
            densities       a density each ( world 2D, with sized balls or ellipses: the mass is the density times the
                            area ). 1 at the start, or the densities of the last stage: `self.densities [ n ]`. Once a
                            cloud has densities, the stages of sized balls and ellipses keep them, fixed unless asked

        None of `radii`, `cov_matrices`, `densities`: balls of ONE radius, `self.radius`, and one density ( the
        densities are forgotten ). `callback( x, loss )` after each iteration, `x` the rows of the shapes. The stage stops
        when the shapes no longer move ( `x_tol` ), or, with `f_tol`, when the loss no longer goes down ( by less than
        `f_tol` times what is left of it above its floor, over 10 iterations: `lbfgs`, "flat" )."""
        md, info = self.model, dict( centers = centers, radii = radii, cov_matrices = cov_matrices, densities = densities )
        if not ( centers or radii or cov_matrices or densities ):
            raise ValueError( "adjust_parameters: say what moves ( centers, radii, cov_matrices, densities )" )
        if not ( self.radius > 0 ):
            raise ValueError( "adjust_parameters: the balls have no radius; set `radius` ( `nominal_radius()`, for instance )" )
        keep = lambda *cols: [ c for c, free in cols if not free ]
        dens = self.densities if self.densities is not None else ( np.ones( self.nb_points ) if densities else None )
        if cov_matrices:
            T = md.tensor
            if self.ellipses is not None:
                x = T( self.ellipses[ :, :5 ] )
                if dens is not None:
                    x = torch.cat( [ x, sized.DENSITY * self.radius * torch.log( T( dens ) )[ :, None ] ], 1 )
            else:
                x = ellipses.from_discs( T( self.points ), T( self.radii ), md, None if dens is None else T( dens ) )
            fixed = keep( ( 0, centers ), ( 1, centers ) ) + ( keep( ( 5, densities ) ) if x.shape[ 1 ] > 5 else [ ] )
            x = self._stage( "adjust_parameters", info, Loss1d( md, ellipses ), x, sigma, background, x_tol, max_iter, fixed, callback, f_tol )
            self.ellipses = x.cpu().numpy()[ :, :5 ]                    # ( the densities: `self.densities` )
        elif ( radii is True or densities or ( radii == "keep" and not np.all( self.radii == self.radius ) )
               or ( radii == "keep" and self.densities is not None ) ):
            x = self._sized( densities )
            fixed = keep( ( 0, centers ), ( 1, centers ), ( 2, radii is True ) ) + ( keep( ( 3, densities ) ) if x.shape[ 1 ] > 3 else [ ] )
            self._stage( "adjust_parameters", info, self._sized_loss( radii is True or bool( densities ) ), x, sigma, background, x_tol, max_iter, fixed, callback, f_tol )
        else:
            self._set( self.points )                                   # ( whatever the shapes were: balls of one radius )
            self._stage( "adjust_parameters", info, self.loss_fn, md.tensor( self.points ), sigma, background, x_tol, max_iter, ( ), callback, f_tol )
        return self

    def fit_polygons( self, nb_vertices = 8, x_tol = 0.001, max_iter = 2000, callback = None ):
        """a stage after the balls ( world 2D ): each shape becomes a STAR POLYGON with a density of its own ( `shapes.polygons` );
        centres and vertices are optimized together. Each ball is taken as the DISC of density 1 of the same mass, and a
        polygon starts as the Laguerre cell of its disc among the others, cut by the disc ( the side between two polygons
        is the common chord of their discs, and a border nothing touches is an arc cut in sides ), then scaled around its
        centre to the area of the disc: the MASS of the ball is kept. `callback( x, loss )` after each iteration, `x` the rows of the polygons.
        The result is `self.polygons [ n, 5 + 2 nb_vertices ]` ( measured against `self.radius` ); `points` and `radii`
        then describe the balls of the same centres and masses."""
        md = self.model
        r = md.tensor( self.radii ) * math.sqrt( md.unit / math.pi )
        x = polygons.from_discs( md.tensor( self.points ), r, self.radius, nb_vertices, math.pi * r * r )
        self.polygons = self._stage( "fit_polygons", dict( nb_vertices = nb_vertices ), Loss1d( md, polygons ), x, 0.0, None, x_tol, max_iter, ( ), callback ).cpu().numpy()
        return self

    def multiscale( self, n_start, n_end, division = 2, x_tol = 0.1, max_iter = 2000 ):
        """the coarse to fine recipe in the NUMBER of balls: about `n_start` random balls and a stage, then, until `n_end`
        balls, `split_items( division )` and a stage again ( `cloud.sizes`: the last division may be partial ). Each stage
        runs at the NOMINAL radius of the cloud ( `nominal_radius`; of the cloud before its division, the room of each
        ball shared between those it becomes )."""
        sizes = cloud.sizes( n_start, n_end, division )
        self.add_random_balls( sizes[ 0 ], 0.0 )
        for n in sizes:
            self.radius = self.nominal_radius()
            if n > self.nb_points:
                self.split_items( division, n = n )
            self.adjust_parameters( centers = True, x_tol = x_tol, max_iter = max_iter )
        return self
