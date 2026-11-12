"""Tomographic reconstruction of a cloud of shapes from a `Sinogram`, in 2D or 3D, by minimizing

    L( points ) = sum_views  W2^2(  diracs of the view ,  image of the projected shapes  )            ( see `loss` )

A shape is a small piece of matter: the object ( continuous, with holes ) is the sum of 1e5 to 1e7 of them, of equal masses.
The state is the current cloud `points [ n, dim ]`; the steps are chainable and each one starts from the current cloud:

    rec = Reconstruction( sino, radius = 0.008 )
    rec.multiscale( 25_000, 100_000, radius = 0.8 )          # or, step by step: rec.random_points( n ), rec.lbfgs(), rec.split()

A STAGE ( `lbfgs` ) moves the shapes until they do not move any more ( `move_tol` ); `multiscale` chains stages, each
shape divided between two of them. `sigma` is a blur applied to both measures, for a stage that starts far
from the object: large, it widens the basin of the minimum.
Every length is in WORLD units ( `radius`, `sigma` ) or relative to the shapes ( `move_tol` ): nothing is counted in pixels
of the sinogram, whose number of bins is a parameter like another.
The cloud is a numpy array between the stages; a stage runs on tensors, on the device of loom ( see `loss.Model` )."""
import time

import numpy as np
import torch

from . import cloud
from .lbfgs import lbfgs
from .loss import Model, make_loss


class Reconstruction:
    """`radius`: of the shapes. `move_tol`: a stage stops when 99.9 % of the shapes have moved by less than that fraction of
    their RADIUS over 10 iterations ( or after `max_iter` ). `background`: both measures are lifted by that fraction of their mean."""

    def __init__( self, sinogram, points = None, *, radius, k = 2, n_per_pixel = 1, background = 1e-3, move_tol = 0.1,
                  max_iter = 2000, seed = 0, view_batch = 8, device = None, recorder = None, verbose = False ):
        self.sinogram, self.dim = sinogram, sinogram.dim
        self.model = Model( sinogram, radius, k, n_per_pixel, view_batch, device )
        self.loss_fn = make_loss( self.model )
        self.background, self.move_tol, self.max_iter, self.verbose = float( background ), float( move_tol ), int( max_iter ), verbose
        self.points = None if points is None else np.array( points, dtype = float )
        self.rng = np.random.default_rng( seed )
        #: one line per stage: sigma, nb_points, radius, loss before / after, iterations, evaluations, why it stopped
        #: ( see `lbfgs` ), seconds
        self.history = [ ]
        #: a `Recorder`, or None
        self.recorder = recorder

    # -- the cloud ---------------------------------------------------------------------------------------------------------

    @property
    def radius( self ):
        """of the shapes ( it may be set: `multiscale` does, from a stage to the next )"""
        return self.model.radius

    @radius.setter
    def radius( self, value ):
        self.model.radius = float( value )

    @property
    def nb_points( self ):
        return 0 if self.points is None else self.points.shape[ 0 ]

    def random_points( self, n, radius = None, height = None ):
        """`n` points drawn uniformly: in the disk ( 2D ) / the cylinder of axis z and height `height` ( 3D ) of radius
        `radius`, or, without `radius`, in the centred ball of a quarter of the detector"""
        if radius is None:
            self.points = cloud.uniform_in_ball( self.rng, n, self.dim, 0.25 * self.sinogram.extent )
        else:
            self.points = cloud.uniform_in_cylinder( self.rng, n, self.dim, radius, height )
        return self

    def split( self, division = 2, n = None ):
        """each shape becomes `division` shapes, a radius around it; with `n`, only `n` of them are kept"""
        self.points = cloud.split( self.rng, self.points, division, self.radius, n )
        return self

    # -- the loss ----------------------------------------------------------------------------------------------------------

    def _pixels( self, sigma ):
        """a blur in the unit of the loss: pixels of the sinogram"""
        return float( sigma ) / self.sinogram.h

    def value_and_grad( self, points = None, sigma = 0.0, background = None ):
        """the loss and its derivative with respect to the points `[ n, dim ]` ( the current cloud by default )"""
        p = self.model.tensor( self.points if points is None else points )
        f, g = self.loss_fn.value_and_grad( p, self._pixels( sigma ), self.background if background is None else float( background ) )
        return f, g.cpu().numpy()

    def image( self, points = None, sigma = 0.0, background = None ):
        """the image of the projected shapes in every view, as the transport sees it ( see `loss.Model.images` )"""
        p = self.model.tensor( self.points if points is None else points )
        bg = self.background if background is None else float( background )
        with torch.no_grad():
            return torch.cat( [ self.model.images( p, self._pixels( sigma ), bg, v ) for v in self.model.batches() ] ).cpu().numpy()

    @property
    def loss( self ):
        return self.history[ -1 ][ "loss" ] if self.history else None

    # -- the steps ---------------------------------------------------------------------------------------------------------

    def lbfgs( self, sigma = 0.0, background = None, move_tol = None, max_iter = None ):
        """one STAGE: L-BFGS on the positions, from the current cloud, at the blur `sigma`"""
        t0, rc = time.perf_counter(), self.recorder
        bg = self.background if background is None else float( background )
        fun = lambda x: self.loss_fn.value_and_grad( x, self._pixels( sigma ), bg )
        x = self.model.tensor( self.points )
        loss0 = fun( x )[ 0 ]
        if rc:
            rc.begin_stage( len( self.history ), sigma, self.radius, sum( h[ "nb_iter" ] for h in self.history ), self.points, loss0 )
        x, loss, nb_iter, nb_eval, stop = lbfgs( fun, x, ( self.move_tol if move_tol is None else move_tol ) * self.radius,
                                           self.max_iter if max_iter is None else max_iter, callback = rc.iteration if rc else None )
        self.points = x.cpu().numpy()
        if rc:
            rc.end_stage( self.points, loss, nb_iter )
        self.history.append( dict( sigma = float( sigma ), nb_points = self.nb_points, radius = self.radius, loss0 = loss0, loss = loss,
                                   nb_iter = nb_iter, nb_eval = nb_eval, stop = stop, seconds = time.perf_counter() - t0 ) )
        if self.verbose:
            print( "  ".join( f"{ k }={ v:.4g}" if isinstance( v, float ) else f"{ k }={ v }" for k, v in self.history[ -1 ].items() ), flush = True )
        return self

    def multiscale( self, n_start, n_end, division = 2, radius = None, height = None ):
        """the coarse to fine recipe in the NUMBER of shapes, from a random start ( see `random_points` ): a stage with
        about `n_start` shapes, then, until `n_end` shapes, each shape divided in `division` and a stage again.
        The shapes are LARGER when they are fewer, so that they hold the same room in all: `self.radius` is their radius at
        the end, `division^( 1 / dim )` times more a stage earlier."""
        sizes, end = cloud.sizes( n_start, n_end, division ), self.radius
        of = lambda n: end * ( n_end / n ) ** ( 1 / self.dim )
        self.radius = of( sizes[ 0 ] )
        self.random_points( sizes[ 0 ], radius, height ).lbfgs()
        for n in sizes[ 1: ]:
            self.split( division, n )                                 # ( each one a radius of its stage around )
            self.radius = of( n )
            self.lbfgs()
        return self
