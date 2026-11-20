import os
import sys

import numpy as np

# shorter names for the line a step prints ( `verbose` ): it fits on a terminal's width
SHORT = { "centers": "ctr", "radii": "rad", "cov_matrices": "cov", "densities": "dens", "sigma": "σ", "x_tol": "xtol",
          "nb_points": "pts", "radius": "r", "nb_iter": "it", "nb_eval": "ev", "fraction": "frac", "removed": "rm",
          "nb_splits": "splits", "division": "div", "noise_ratio": "noise",
          # the steps themselves
          "adjust_parameters": "adjust", "add_random_balls": "add_balls", "remove_in_air": "remove_air" }


def _coloured( ) -> bool:
    if os.environ.get( "NO_COLOR" ):
        return False
    if os.environ.get( "FORCE_COLOR" ) or os.environ.get( "CLICOLOR_FORCE" ):
        return True
    try:
        return sys.stdout.isatty()
    except ( AttributeError, ValueError ):
        return False


def step_groups( step: dict ):
    """`( head, groups )` of the line a step prints: the step, then its parameters, its loss, its counts, its time
    -- the names in grey on a terminal, the values as they are; True / False as 1 / 0"""
    grey = ( lambda s: f"\033[2m{ s }\033[0m" ) if _coloured() else ( lambda s: s )
    value = lambda v: ( "1" if v else "0" ) if isinstance( v, bool ) else f"{ v:.3g}" if isinstance( v, float ) else str( v )
    step = dict( step )
    head = str( step.pop( "step", "" ) )
    head = SHORT.get( head, head )
    seconds, stop = step.pop( "seconds", None ), step.pop( "stop", None )
    loss0, loss = step.pop( "loss0", None ), step.pop( "loss", None )
    counts = { k: step.pop( k ) for k in ( "nb_iter", "nb_eval" ) if k in step }
    pairs = lambda d: [ f"{ grey( SHORT.get( k, k ) + '=' ) }{ value( v ) }" for k, v in d.items() ]
    groups = [ pairs( step ) ]
    if loss is not None:
        groups.append( ( [ f"{ grey( 'loss' ) } { value( float( loss0 ) ) }", f"{ grey( '→' ) } { value( float( loss ) ) }" ]
                         if loss0 is not None else [ f"{ grey( 'loss' ) } { value( float( loss ) ) }" ] ) )
    if counts or stop:
        groups.append( pairs( counts ) + ( [ str( stop ) ] if stop else [ ] ) )
    if seconds is not None:
        groups.append( [ f"{ float( seconds ):.1f}{ grey( ' s' ) }" ] )
    return head, [ g for g in groups if g ]


def step_line( step: dict ) -> str:
    """`adjust  ctr=1 rad=0 cov=0 dens=0 σ=0 xtol=0.1 pts=5000 r=0.0166  loss 9.39 → 0.00231  it=360 ev=388 still  11.6 s`.
    Under errand, a row ( `errand.row` ): its pieces aligned with those of the steps before"""
    head, groups = step_groups( step )
    try:
        from errand.rows import text
    except ImportError:
        return "  ".join( [ head ] + [ " ".join( g ) for g in groups ] )
    return text( head, *groups )


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
        fields       `{ name: [ n ] }` float32, what the views ask of each shape ( `loss.fields`: balls, sized balls; with
                     `fields`, the default ): the page colours the shapes with them

    `verbose`: each step printed when it ends. A `Reconstruction` sets `dim` and `floor` ( of its loss ).

    `store = "foam.sviz"`: the frames go to a SCENE STORE ( `sdot.viz.store`, errand ADR 0017 ) as they are made, instead of
    staying in memory for `write_html`: each frame is drawn there at once ( the page of `write_html`, read frame by frame
    by `sviz open`, while the run goes on ), and `frames` keeps only what the curves need ( no points, radii, drawing,
    fields; `median_radius` instead ). `page`: the options of `write_html` ( title, truth_edges, truth_points, images,
    truth_name, field, curves, fps ), known from the start. `on_frame`: called each time the store says it changed ( say,
    `lambda: errand.view( store )` ). `close()` writes the last frame; it is called at the end of the process if not
    before ( `atexit` ), and a second call does nothing."""

    def __init__( self, every = None, metrics = None, verbose = False, fields = True, store = None, page = None, on_frame = None ):
        self.every, self.metrics, self.verbose = None if every is None else max( 1, int( every ) ), dict( metrics or { } ), verbose
        self.fields = fields
        self.steps, self.frames, self.losses = [ ], [ ], [ ]
        self.dim, self.floor = 2, 0.0
        self.store, self.page, self.on_frame = store, dict( page or { } ), on_frame
        self._viz = None
        if store is not None:                                          # the store closed at the end of the run, if not before
            import atexit
            atexit.register( self.close )

    @property
    def stages( self ):
        return [ s for s in self.steps if "nb_iter" in s ]

    @property
    def iteration( self ):
        """the number of iterations of all the stages so far"""
        return sum( s[ "nb_iter" ] for s in self.stages )

    # -- what a reconstruction tells ---------------------------------------------------------------------------------------

    def step( self, step, snapshot = None, **info ):
        """a step that changes the cloud without a stage ( `add_random_balls`, `split_items`, `remove_isolated` ). With
        `snapshot` ( as for a stage ) and once a stage has run, a frame of the cloud after it, at the same iteration
        ( its loss: `info[ "loss" ]` )"""
        self._end( dict( step = step, **info ) )
        if snapshot is not None and self.frames:
            self._add( snapshot, info.get( "loss", self.frames[ -1 ][ "loss" ] ), self.iteration - self._it0 )

    def begin_stage( self, step, info, snapshot, loss ):
        """`info`: the parameters of the stage ( `sigma` among them ), its `nb_points` and `radius`; `snapshot()`:
        `( points, radii, drawing, fields )`, called only for a frame"""
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
            print( step_line( step ), flush = True )

    def _add( self, snapshot, loss, it ):
        if not self.every:
            return
        points, radii, drawing, fields = snapshot()
        drawing = { k: tuple( np.asarray( a, dtype = np.float32 if a.dtype.kind == "f" else np.int32 ) for a in v ) if isinstance( v, tuple )
                       else np.asarray( v, dtype = np.float32 ) for k, v in drawing.items() }
        frame = dict( iteration = self._it0 + it, points = points.astype( "float32" ), radii = radii.astype( "float32" ), drawing = drawing,
                      fields = { k: np.asarray( v, dtype = np.float32 ) for k, v in fields.items() }, loss = float( loss ), metrics = { name: float( f( points, radii ) ) for name, f in self.metrics.items() },
                      **dict( self._info, nb_points = len( points ) ) )
        if self.store is None:
            self.frames.append( frame )
        else:
            self._to_store( frame )

    # -- the store ---------------------------------------------------------------------------------------------------------

    def _to_store( self, frame ):
        """the frame drawn in the store at once, and forgotten; the curves given again, longer"""
        opts = dict( self.page )
        field = opts.get( "field", "dissonance" )
        if self._viz is None:
            self._viz = self._page_viz( frame[ "iteration" ], store = self.store, on_frame = self.on_frame, **dict( opts, field = None ) )
        else:
            self._viz.new_frame( frame[ "iteration" ] )
        if self._viz.field is None and field in frame[ "fields" ]:          # ( the field shown at the start, once a frame has it )
            self._viz.field = field
        self._draw( self._viz, frame )
        self.frames.append( dict( { k: v for k, v in frame.items() if k not in ( "points", "radii", "drawing", "fields" ) },
                                  median_radius = float( np.median( frame[ "radii" ] ) ) if len( frame[ "radii" ] ) else 0.0 ) )
        self._curves( self._viz.clear_curves(), self.frames, opts.get( "curves", ( "loss", ) ) )

    def close( self, html = None ):
        """the store: its last frame written, and said complete ( nothing to do without a store ). `html`: then the
        self-contained page of it too ( `sviz export --html`, a link to it said: `OUTPUT: file://...` )"""
        if self._viz is not None:
            self._curves( self._viz.clear_curves(), self.frames, self.page.get( "curves", ( "loss", ) ) )    # ( the losses after the last frame too )
            self._viz.close()
            self._viz = None
            if html is not None:
                from sdot.viz.sviz import export_html
                export_html( self.store, html )

    # -- the page ----------------------------------------------------------------------------------------------------------

    def _page_viz( self, first_iteration, title = None, truth_edges = None, truth_points = None, images = None, truth_name = "truth",
                   field = "dissonance", curves = ( "loss", ), fps = 6.0, store = None, on_frame = None ):
        """the Visualizer of the page: its options, the images and the truth ( in every frame ), no frame drawn"""
        from sdot.viz.Visualizer import Visualizer
        from .viz import TRUTH
        dim = self.dim
        viz = Visualizer( title = title or f"otrec { dim }D", frame_axis = "iteration", frame_value = first_iteration, fps = fps, flat_points = dim == 2,
                          compact = True, field = field, store = store, on_frame = on_frame )
        if truth_edges is not None:
            truth_edges = ( np.asarray( truth_edges[ 0 ], dtype = np.float32 ), np.asarray( truth_edges[ 1 ] ) )
        if truth_points is not None:
            truth_points = np.asarray( truth_points, dtype = np.float32 )
        for name, ( values, extent ) in ( images or { } ).items():
            viz.add_layer( name, visible = False )
            viz.add_image( values, ( -extent / 2, -extent / 2 ), ( extent / 2, extent / 2 ), layer = name )
        if truth_points is not None:                          # ( the truth: in every frame, stored once )
            viz.add_points( truth_points, color = TRUTH, opacity = 0.35, shared = True, layer = truth_name )
        if truth_edges is not None:
            viz.add_edges( truth_edges[ 0 ], truth_edges[ 1 ], color = TRUTH, shared = True, layer = truth_name )
        return viz

    def _draw( self, viz, fr ):
        """the shapes of a frame, as they are"""
        from .viz import CLOUD
        if "matrices" in fr[ "drawing" ]:
            viz.add_points( fr[ "points" ], matrices = fr[ "drawing" ][ "matrices" ], color = CLOUD, opacity = 0.6, fields = fr.get( "fields" ) )
        elif "faces" in fr[ "drawing" ]:
            viz.add_faces( *fr[ "drawing" ][ "faces" ], color = CLOUD, opacity = 0.5 )
        else:
            viz.add_points( fr[ "points" ], radius = fr[ "radii" ], color = CLOUD, opacity = 0.6 if self.dim == 2 else 1.0, fields = fr.get( "fields" ) )

    def _curves( self, viz, frames, curves ):
        """the loss at every iteration, and, per frame, the metrics, the blur, the number of shapes, their median radius"""
        shown = lambda name: name in curves or ( name.startswith( "loss" ) and "loss" in curves )
        its = [ fr[ "iteration" ] for fr in frames ]
        # the loss ends on a floor it cannot go under ( `Loss1d.floor` ): on a log scale, only what is ABOVE it shows the convergence
        floor = self.floor
        name = "loss above its floor" if floor else "loss"
        viz.add_curve( name, [ f - floor for _, f in self.losses ], [ it for it, _ in self.losses ], log_y = True, visible = shown( name ) )
        for name in frames[ 0 ][ "metrics" ]:
            log = name.endswith( "( log )" )
            viz.add_curve( name[ : -len( "( log )" ) ].strip() if log else name, [ fr[ "metrics" ][ name ] for fr in frames ], its, log_y = log, visible = shown( name ) )
        viz.add_curve( "blur", [ fr[ "sigma" ] for fr in frames ], its, visible = shown( "blur" ) )
        viz.add_curve( "number of shapes", [ fr[ "nb_points" ] for fr in frames ], its, visible = shown( "number of shapes" ) )
        viz.add_curve( "median radius of the shapes", [ fr[ "median_radius" ] if "median_radius" in fr else float( np.median( fr[ "radii" ] ) ) for fr in frames ],
                       its, visible = shown( "median radius of the shapes" ) )

    def write_html( self, out_path, title = None, truth_edges = None, truth_points = None, images = None, truth_name = "truth", field = "dissonance",
                    curves = ( "loss", ), fps = 6.0 ):
        """the frames as one standalone HTML page ( `sdot.Visualizer` ): one frame per recorded iteration ( the play bar,
        the arrows ), the shapes as they are ( balls at their radius, ellipses, polygons ), over the truth when it is given;
        and, in the panel of the curves, the loss ( above its floor ), the metrics, the blur, the number of shapes, with a
        cursor at the displayed frame ( a click in a chart goes to that iteration ). `truth_edges`: `( positions, edges )`,
        the outline of the object ( 2D: see `viz.contours` ); `truth_points`: `[ m, dim ]`, a sample of it ( 3D ), both in
        a layer `truth_name` of the menu. `images`: `{ name: ( values [ nx, ny ], extent ) }`, each over the centred square
        of side `extent`, under the shapes, in a layer of its own, hidden at the start ( a filtered back projection, say ).
        The shapes can be coloured by the FIELDS of the frames ( see `loss.fields` ), `field` at the start ( None: their
        colour ). The charts are windows the page shows or hides: `curves` names those shown at the start ( "loss", a
        metric, "blur", "number of shapes", "median radius of the shapes" ). A metric whose name ends with "( log )" is
        drawn on a log scale."""
        frames = self.frames
        if self.store is not None:
            raise ValueError( f"Recorder: the frames went to the store { self.store } ( `close()` it; `sviz export --html` makes a page of it )" )
        assert frames, "no frame: give the recorder `every`"
        viz = self._page_viz( frames[ 0 ][ "iteration" ], title = title, truth_edges = truth_edges, truth_points = truth_points, images = images,
                              truth_name = truth_name, fps = fps, field = field if any( field in fr.get( "fields", { } ) for fr in frames ) else None )
        for f, fr in enumerate( frames ):
            if f:
                viz.new_frame( fr[ "iteration" ] )
            self._draw( viz, fr )
        self._curves( viz, frames, curves )
        viz.write_html( str( out_path ) )
        return viz
