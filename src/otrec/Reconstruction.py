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
    rec.adjust_parameters( centers = True, x_tol = 0.1 )
    rec.adjust_parameters( centers = True, cov_matrices = True, x_tol = 0.01 )  # ellipses: a covariance each ( 2D )

`adjust_parameters` says what moves: the centres, a radius each ( `radii`, 2D ), a covariance each ( `cov_matrices`,
2D: ellipses with the profile of the balls ). `fit_polygons` makes star polygons of density 1 ( 2D ). `multiscale` is a
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
    history ( `Recorder()` by default: the steps and the losses, no frames ). Random draws take `rng` ( seeded with 0 )."""

    def __init__( self, sinogram, points = None, *, radius = None, k = 2, n_per_pixel = 1, background = 1e-3, view_batch = 8,
                  device = None, max_ratio = 4.0, recorder = None ):
        self.sinogram, self.dim = sinogram, sinogram.dim
        self.model = Model( sinogram, radius, k, n_per_pixel, view_batch, device, max_ratio )
        self.loss_fn = make_loss( self.model )
        self.background = float( background )
        self.recorder = Recorder() if recorder is None else recorder
        self.recorder.dim, self.recorder.floor = self.dim, self.loss_fn.floor
        self.rng = np.random.default_rng( 0 )
        self.points = self.radii = self.ellipses = None
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
            self._set( self.points )

    @property
    def nb_points( self ):
        return 0 if self.points is None else self.points.shape[ 0 ]

    def _set( self, points, radii = None ):
        """the cloud: balls ( of `self.radius`, or of `radii` ); the ellipses are forgotten"""
        self.points, self.radii, self.ellipses = points, np.full( len( points ), self.radius ) if radii is None else radii, None
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
        self._set( new )
        self.recorder.step( "split_items", division = division, noise_ratio = noise_ratio, nb_points = self.nb_points, radius = self.radius )
        return self

    def nominal_radius( self ):
        """the radius at which balls spaced as the cloud is FILL the matter: each centre stands for a room `spacing^dim`
        ( `cloud.spacing` ), and a ball of that mass and of the density of the matter at its centre has that radius.
        ( 2D: checked on packings; 3D: the same formula, not calibrated. )"""
        return cloud.spacing( self.points ) / self.model.unit ** ( 1 / self.dim )

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

    @property
    def loss( self ):
        """of the last stage"""
        stages = self.recorder.stages
        return stages[ -1 ][ "loss" ] if stages else None

    # -- the steps ---------------------------------------------------------------------------------------------------------

    def _sized( self ):
        return sized.pack( self.model.tensor( self.points ), self.model.tensor( self.radii ), self.radius )

    def _stage( self, step, info, loss_fn, x, sigma, background, x_tol, max_iter, frozen = 0, callback = None ):
        """L-BFGS on `x`, the rows of a kind of shapes ( `loss_fn.kind` ), its first `frozen` columns kept; sets the cloud
        from the result, and returns it"""
        t0, rc, md = time.perf_counter(), self.recorder, self.model
        bg = self.background if background is None else float( background )
        def fun( x ):
            f, g = loss_fn.value_and_grad( x, self._pixels( sigma ), bg )
            if frozen:
                g[ :, :frozen ] = 0
            return f, g
        kind = getattr( loss_fn, "kind", balls )
        def cloud_of( x ):                                             # `( points, radii )`
            with torch.no_grad():
                return tuple( t.cpu().numpy() for t in kind.discs( x, md ) )
        def snapshot( x ):                                             # what a frame keeps: `( points, radii, drawing )`
            return ( *cloud_of( x ), kind.drawing( x, md ) if hasattr( kind, "drawing" ) else { } )
        def each( x, f ):
            rc.iteration_done( lambda: snapshot( x ), f )
            if callback:
                callback( x, f )
        loss0 = fun( x )[ 0 ]
        rc.begin_stage( step, dict( info, sigma = float( sigma ), x_tol = x_tol, nb_points = self.nb_points, radius = self.radius ), lambda: snapshot( x ), loss0 )
        x, loss, nb_iter, nb_eval, stop = lbfgs( fun, x, x_tol * self.radius, max_iter, callback = each )
        self._set( *cloud_of( x ) )
        rc.end_stage( lambda: snapshot( x ), loss0 = loss0, loss = loss, nb_iter = nb_iter, nb_eval = nb_eval, stop = stop, seconds = time.perf_counter() - t0 )
        return x

    def adjust_parameters( self, centers = False, radii = False, cov_matrices = False, x_tol = 0.1, max_iter = 2000, sigma = 0.0, background = None, callback = None ):
        """a STAGE: L-BFGS on what is asked ( nothing moves unless it is said ), from the current cloud, at the blur `sigma`:

            centers         the centres
            radii           a radius each ( world 2D, `shapes.sized`; the mass follows the radius )
            cov_matrices    a covariance each ( world 2D, `shapes.ellipses`: the profile of the balls, stretched; the mass
                            follows the area ). Round at the start, or the ellipses of the last stage. The result is
                            `self.ellipses [ n, 5 ]`; `points` and `radii` then describe the discs of the same areas

        Neither `radii` nor `cov_matrices`: balls of ONE radius, `self.radius`. `callback( x, loss )` after each iteration,
        `x` the rows of the shapes."""
        md, info = self.model, dict( centers = centers, radii = radii, cov_matrices = cov_matrices )
        if not ( centers or radii or cov_matrices ):
            raise ValueError( "adjust_parameters: say what moves ( centers, radii, cov_matrices )" )
        if cov_matrices:
            x = md.tensor( self.ellipses ) if self.ellipses is not None else ellipses.from_discs( md.tensor( self.points ), md.tensor( self.radii ), md )
            x = self._stage( "adjust_parameters", info, Loss1d( md, ellipses ), x, sigma, background, x_tol, max_iter, 0 if centers else 2, callback )
            self.ellipses = x.cpu().numpy()
        elif radii:
            self._stage( "adjust_parameters", info, Loss1d( md, sized ), self._sized(), sigma, background, x_tol, max_iter, 0 if centers else 2, callback )
        else:
            self._set( self.points )                                   # ( whatever the shapes were: balls of one radius )
            self._stage( "adjust_parameters", info, self.loss_fn, md.tensor( self.points ), sigma, background, x_tol, max_iter, 0, callback )
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
        self.polygons = self._stage( "fit_polygons", dict( nb_vertices = nb_vertices ), Loss1d( md, polygons ), x, 0.0, None, x_tol, max_iter, 0, callback ).cpu().numpy()
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
