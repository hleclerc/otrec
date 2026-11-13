class Recorder:
    """Keeps FRAMES of a reconstruction, for `viz.export_html`: one every `every` iterations of each stage, plus the cloud
    the stage starts from and ends at. A frame is a dict:

        iteration    counted over all the stages          stage, sigma, radius ( the reference of the stage ), nb_points, loss
        points       `[ n, dim ]`, radii `[ n ]`: float32
        metrics      `{ name: value }`, each function of `metrics` evaluated on `( points, radii )`

    and `losses`: `( iteration, loss )` at EVERY iteration."""

    def __init__( self, every = 1, metrics = None ):
        self.every, self.metrics = max( 1, int( every ) ), dict( metrics or { } )
        self.frames, self.losses = [ ], [ ]

    def begin_stage( self, stage, sigma, radius, iteration, points, radii, loss ):
        self._info = dict( stage = stage, sigma = float( sigma ), radius = float( radius ), nb_points = len( points ) )
        self._it0, self._count = iteration, 0
        self.losses.append( ( iteration, float( loss ) ) )
        self._add( points, radii, loss, 0 )

    def iteration( self, shapes, loss ):
        """after each iteration of the stage; `shapes()`: `( points, radii )`"""
        self._count += 1
        self.losses.append( ( self._it0 + self._count, float( loss ) ) )
        if self._count % self.every == 0:
            self._add( *shapes(), loss, self._count )

    def end_stage( self, points, radii, loss, nb_iter ):
        if self.frames[ -1 ][ "iteration" ] != self._it0 + nb_iter:
            self._add( points, radii, loss, nb_iter )

    def _add( self, points, radii, loss, it ):
        self.frames.append( dict( iteration = self._it0 + it, points = points.astype( "float32" ), radii = radii.astype( "float32" ), loss = float( loss ),
                                  metrics = { name: float( f( points, radii ) ) for name, f in self.metrics.items() }, **self._info ) )
