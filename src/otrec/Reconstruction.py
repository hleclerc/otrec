"""Tomographic reconstruction of a cloud of shapes from a `Sinogram`, in 2D or 3D, by minimizing

    L( shapes ) = sum_views  W2^2(  diracs of the view ,  image of the projected shapes  )            ( see `loss` )

A shape is a small piece of matter: the object ( continuous, with holes ) is the sum of 1e5 to 1e7 of them.
The state is the current cloud, `points [ n, dim ]` and `radii [ n ]`, the mass of a shape following its size
( `radius^dim` ); the steps are chainable and each one starts from the current cloud:

    rec = Reconstruction( sino )
    rec.multiscale( 25_000, 100_000, radius = 0.8 )          # balls of ONE radius, the nominal one: where the matter is
    rec.to_polygons()                                        # then polygons of density 1 ( 2D )

A STAGE of balls ( `lbfgs` ) moves the centres until they do not move any more ( `move_tol` ); `multiscale` chains
stages, each shape divided between two of them. The radius is not a guess: it is the NOMINAL one, read on the spacing of
the cloud ( `nominal_radius` ). `sigma` is a blur applied to both measures, for a stage that starts far from the object.
Every length is in WORLD units ( `radius`, `sigma` ) or relative to the shapes ( `move_tol` ): nothing is counted in pixels
of the sinogram, whose number of bins is a parameter like another. Whatever the stage, the shapes stay inside what every
view sees ( `shapes.balls.inside` ).
The cloud is numpy arrays between the stages; a stage runs on tensors, on the device of loom ( see `loss.Model` )."""
import math
import time

import numpy as np
import torch

from . import cloud
from .lbfgs import lbfgs
from .loss import Loss1d, Model, make_loss
from .shapes import balls, polygons, sized


class Reconstruction:
    """`radius`: of the balls ( `multiscale` sets it by itself ); polygons stay within `max_ratio` of it.
    `move_tol`: a stage stops when 99.9 % of the shapes have moved by less than that fraction of the RADIUS over 10
    iterations ( or after `max_iter` ). `background`: both measures are lifted by that fraction of their mean."""

    def __init__( self, sinogram, points = None, *, radius = None, k = 2, n_per_pixel = 1, background = 1e-3, move_tol = 0.1,
                  max_iter = 2000, seed = 0, view_batch = 8, device = None, max_ratio = 4.0, recorder = None, verbose = False ):
        self.sinogram, self.dim = sinogram, sinogram.dim
        self.model = Model( sinogram, radius, k, n_per_pixel, view_batch, device, max_ratio )
        self.loss_fn = make_loss( self.model )
        self.background, self.move_tol, self.max_iter, self.verbose = float( background ), float( move_tol ), int( max_iter ), verbose
        self.points = self.radii = None
        if points is not None:
            self._set( np.array( points, dtype = float ) )
        self.rng = np.random.default_rng( seed )
        #: one line per stage: shapes, sigma, nb_points, radius, loss before / after, iterations, evaluations, why it
        #: stopped ( see `lbfgs` ), seconds
        self.history = [ ]
        #: a `Recorder`, or None
        self.recorder = recorder

    # -- the cloud ---------------------------------------------------------------------------------------------------------

    @property
    def radius( self ):
        """of the balls; set, it makes every shape a ball of that radius"""
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
        self.points, self.radii = points, np.full( len( points ), self.radius ) if radii is None else radii
        return self

    def random_points( self, n, radius = None, height = None ):
        """`n` balls drawn uniformly: in the disk ( 2D ) / the cylinder of axis z and height `height` ( 3D ) of radius
        `radius`, or, without `radius`, in the centred ball of a quarter of the detector"""
        if radius is None:
            return self._set( cloud.uniform_in_ball( self.rng, n, self.dim, 0.25 * self.sinogram.extent ) )
        return self._set( cloud.uniform_in_cylinder( self.rng, n, self.dim, radius, height ) )

    def split( self, division = 2, n = None ):
        """each ball becomes `division` balls, a radius around it; with `n`, only `n` of them are kept"""
        return self._set( cloud.split( self.rng, self.points, division, self.radius, n ) )

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
        return self.history[ -1 ][ "loss" ] if self.history else None

    # -- the steps ---------------------------------------------------------------------------------------------------------

    def _sized( self ):
        return sized.pack( self.model.tensor( self.points ), self.model.tensor( self.radii ), self.radius )

    def _stage( self, name, loss_fn, x, sigma, background, move_tol, max_iter, callback = None ):
        """L-BFGS on `x`, the rows of a kind of shapes ( `loss_fn.kind` ); sets the cloud from the result, and returns it"""
        t0, rc, md = time.perf_counter(), self.recorder, self.model
        bg = self.background if background is None else float( background )
        fun = lambda x: loss_fn.value_and_grad( x, self._pixels( sigma ), bg )
        def cloud_of( x ):                                             # `( points, radii )`
            with torch.no_grad():
                return tuple( t.cpu().numpy() for t in getattr( loss_fn, "kind", balls ).discs( x, md ) )
        def each( x, f ):
            if rc:
                rc.iteration( lambda: cloud_of( x ), f )
            if callback:
                callback( x, f )
        loss0 = fun( x )[ 0 ]
        if rc:
            rc.begin_stage( len( self.history ), sigma, self.radius, sum( h[ "nb_iter" ] for h in self.history ), self.points, self.radii, loss0 )
        x, loss, nb_iter, nb_eval, stop = lbfgs( fun, x, ( self.move_tol if move_tol is None else move_tol ) * self.radius,
                 self.max_iter if max_iter is None else max_iter, callback = each if rc or callback else None )
        self._set( *cloud_of( x ) )
        if rc:
            rc.end_stage( self.points, self.radii, loss, nb_iter )
        self.history.append( dict( shapes = name, sigma = float( sigma ), nb_points = self.nb_points, radius = self.radius, loss0 = loss0, loss = loss,
                                   nb_iter = nb_iter, nb_eval = nb_eval, stop = stop, seconds = time.perf_counter() - t0 ) )
        if self.verbose:
            print( "  ".join( f"{ k }={ v:.4g}" if isinstance( v, float ) else f"{ k }={ v }" for k, v in self.history[ -1 ].items() ), flush = True )
        return x

    def lbfgs( self, sigma = 0.0, background = None, move_tol = None, max_iter = None ):
        """one STAGE of balls of radius `self.radius`: L-BFGS on the centres, from the current ones, at the blur `sigma`"""
        self.radius = self.radius                                      # ( whatever the shapes were )
        self._stage( "balls", self.loss_fn, self.model.tensor( self.points ), sigma, background, move_tol, max_iter )
        return self

    def resize( self, move_tol = None, max_iter = None ):
        """a stage of balls with a radius EACH ( world 2D, `shapes.sized` ): centres and radii together, from the current cloud"""
        self._stage( "sized balls", Loss1d( self.model, sized ), self._sized(), 0.0, None, move_tol, max_iter )
        return self

    def to_polygons( self, nb_vertices = 8, move_tol = 0.001, max_iter = None, callback = None ):
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
        self.polygons = self._stage( "polygons", Loss1d( md, polygons ), x, 0.0, None, move_tol, max_iter, callback ).cpu().numpy()
        return self

    def multiscale( self, n_start, n_end, division = 2, radius = None, height = None ):
        """the coarse to fine recipe in the NUMBER of balls, from a random start ( see `random_points` ): a stage with
        about `n_start` balls, then, until `n_end` balls, each ball divided in `division` and a stage again.
        Each stage runs at the NOMINAL radius of the cloud it starts from ( `nominal_radius`; of the cloud before its
        division, for the room of each ball shared between those it becomes )."""
        sizes = cloud.sizes( n_start, n_end, division )
        self.random_points( sizes[ 0 ], radius, height )
        for n in sizes:
            self.radius = self.nominal_radius() * ( self.nb_points / n ) ** ( 1 / self.dim )
            if n > self.nb_points:
                self.split( division, n )
            self.lbfgs()
        return self
