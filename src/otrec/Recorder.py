import numpy as np


class Recorder:
    """The HISTORY of a reconstruction: every step it was asked for, with its parameters, what each stage gave, and the
    page that shows it ( `write_html` ).

        steps     one dict per step, in order: `step` ( "add_random_balls", "split_items", "adjust_parameters", ... ),
                  its parameters, `nb_points`, `radius`; a STAGE ( an L-BFGS ) adds `loss0`, `loss`, `nb_iter`, `nb_eval`,
                  `stop` ( see `lbfgs` ), `seconds`
        stages    the steps that are stages
        losses    `( iteration, loss )` at EVERY iteration, counted over all the stages

    With `every`, FRAMES too: one every `every` iterations of each stage, plus the cloud the stage starts from and ends
    at. A frame is a dict:

        iteration    counted over all the stages          stage, sigma, radius ( the reference of the stage ), nb_points, loss
        points       `[ n, dim ]`, radii `[ n ]`: float32 ( the discs of the shapes )
        drawing      how the shapes are drawn when they are not balls: `{ "matrices": [ n, 2, 2 ] }` ( ellipses ),
                     `{ "faces": ( positions, triangles ) }` ( polygons ), `{ }` ( balls: the discs )
        metrics      `{ name: value }`, each function of `metrics` evaluated on `( points, radii )`

    `verbose`: each step printed when it ends. A `Reconstruction` sets `dim` and `floor` ( of its loss )."""

    def __init__( self, every = None, metrics = None, verbose = False ):
        self.every, self.metrics, self.verbose = None if every is None else max( 1, int( every ) ), dict( metrics or { } ), verbose
        self.steps, self.frames, self.losses = [ ], [ ], [ ]
        self.dim, self.floor = 2, 0.0

    @property
    def stages( self ):
        return [ s for s in self.steps if "loss" in s ]

    @property
    def iteration( self ):
        """the number of iterations of all the stages so far"""
        return sum( s[ "nb_iter" ] for s in self.stages )

    # -- what a reconstruction tells ---------------------------------------------------------------------------------------

    def step( self, step, **info ):
        """a step that changes the cloud without a stage ( `add_random_balls`, `split_items` )"""
        self._end( dict( step = step, **info ) )

    def begin_stage( self, step, info, snapshot, loss ):
        """`info`: the parameters of the stage ( `sigma` among them ), its `nb_points` and `radius`; `snapshot()`:
        `( points, radii, drawing )`, called only for a frame"""
        self._stage = dict( step = step, **info )
        self._it0, self._count = self.iteration, 0
        self._info = dict( stage = len( self.stages ), sigma = float( info.get( "sigma", 0.0 ) ), radius = float( info[ "radius" ] ), nb_points = info[ "nb_points" ] )
        self.losses.append( ( self._it0, float( loss ) ) )
        self._add( snapshot, loss, 0 )

    def iteration_done( self, snapshot, loss ):
        """after each iteration of the stage"""
        self._count += 1
        self.losses.append( ( self._it0 + self._count, float( loss ) ) )
        if self.every and self._count % self.every == 0:
            self._add( snapshot, loss, self._count )

    def end_stage( self, snapshot, **result ):
        """`result`: loss0, loss, nb_iter, nb_eval, stop, seconds"""
        if self.every and self.frames[ -1 ][ "iteration" ] != self._it0 + result[ "nb_iter" ]:
            self._add( snapshot, result[ "loss" ], result[ "nb_iter" ] )
        self._end( dict( self._stage, **result ) )

    def _end( self, step ):
        self.steps.append( step )
        if self.verbose:
            print( "  ".join( f"{ k }={ v:.4g}" if isinstance( v, float ) else f"{ k }={ v }" for k, v in step.items() ), flush = True )

    def _add( self, snapshot, loss, it ):
        if not self.every:
            return
        points, radii, drawing = snapshot()
        drawing = { k: tuple( np.asarray( a, dtype = np.float32 if a.dtype.kind == "f" else np.int32 ) for a in v ) if isinstance( v, tuple )
                       else np.asarray( v, dtype = np.float32 ) for k, v in drawing.items() }
        self.frames.append( dict( iteration = self._it0 + it, points = points.astype( "float32" ), radii = radii.astype( "float32" ), drawing = drawing,
                                  loss = float( loss ), metrics = { name: float( f( points, radii ) ) for name, f in self.metrics.items() }, **self._info ) )

    # -- the page ----------------------------------------------------------------------------------------------------------

    def write_html( self, out_path, title = None, truth_edges = None, truth_points = None, fps = 6.0 ):
        """the frames as one standalone HTML page ( `sdot.Visualizer` ): one frame per recorded iteration ( the play bar,
        the arrows ), the shapes as they are ( balls at their radius, ellipses, polygons ), over the truth when it is given;
        and, in the panel of the curves, the loss ( above its floor ), the metrics, the blur, the number of shapes, with a
        cursor at the displayed frame ( a click in a chart goes to that iteration ). `truth_edges`: `( positions, edges )`,
        the outline of the object ( 2D: see `viz.contours` ); `truth_points`: `[ m, dim ]`, a sample of it ( 3D ). A metric
        whose name ends with "( log )" is drawn on a log scale."""
        from sdot.viz.Visualizer import Visualizer
        from .viz import CLOUD, TRUTH
        frames, dim = self.frames, self.dim
        assert frames, "no frame: give the recorder `every`"
        its = [ fr[ "iteration" ] for fr in frames ]
        viz = Visualizer( title = title or f"otrec { dim }D", frame_axis = "iteration", frame_value = its[ 0 ], fps = fps, flat_points = dim == 2 )
        if truth_edges is not None:
            truth_edges = ( np.asarray( truth_edges[ 0 ], dtype = np.float32 ), np.asarray( truth_edges[ 1 ] ) )
        if truth_points is not None:
            truth_points = np.asarray( truth_points, dtype = np.float32 )
        if truth_points is not None:                          # ( the truth: in every frame, stored once )
            viz.add_points( truth_points, color = TRUTH, opacity = 0.35, shared = True )
        if truth_edges is not None:
            viz.add_edges( truth_edges[ 0 ], truth_edges[ 1 ], color = TRUTH, shared = True )
        for f, fr in enumerate( frames ):
            if f:
                viz.new_frame( fr[ "iteration" ] )
            if "matrices" in fr[ "drawing" ]:
                viz.add_points( fr[ "points" ], matrices = fr[ "drawing" ][ "matrices" ], color = CLOUD, opacity = 0.6 )
            elif "faces" in fr[ "drawing" ]:
                viz.add_faces( *fr[ "drawing" ][ "faces" ], color = CLOUD, opacity = 0.5 )
            else:
                viz.add_points( fr[ "points" ], radius = fr[ "radii" ], color = CLOUD, opacity = 0.6 if dim == 2 else 1.0 )
        # the loss ends on a floor it cannot go under ( `Loss1d.floor` ): on a log scale, only what is ABOVE it shows the convergence
        floor = self.floor
        viz.add_curve( "loss above its floor" if floor else "loss", [ f - floor for _, f in self.losses ], [ it for it, _ in self.losses ], log_y = True )
        for name in frames[ 0 ][ "metrics" ]:
            log = name.endswith( "( log )" )
            viz.add_curve( name[ : -len( "( log )" ) ].strip() if log else name, [ fr[ "metrics" ][ name ] for fr in frames ], its, log_y = log )
        viz.add_curve( "blur", [ fr[ "sigma" ] for fr in frames ], its )
        viz.add_curve( "number of shapes", [ fr[ "nb_points" ] for fr in frames ], its )
        viz.add_curve( "median radius of the shapes", [ float( np.median( fr[ "radii" ] ) ) for fr in frames ], its )
        viz.write_html( str( out_path ) )
        return viz
