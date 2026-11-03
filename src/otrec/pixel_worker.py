"""A process per card for the 2D loss of `Reconstruction3d` ( `pixel_loss` ): each one sees ONE card ( `CUDA_VISIBLE_DEVICES` ),
holds the `PixelTransport2d` of its views ( with their dual weights: the warm starts stay there ), and answers, for the
points it is sent, the sum of the costs of its views and their gradient. loom keeps one default card per process: views
spread over cards are processes, not threads.

Nothing here imports jax at the top: the child must restrict its card before jax sees any.
"""


def serve( conn, card, values, bases, offsets, detector, setting, threads = None ):
    """the loop of a worker: `( "eval", x, radius )` -> `( "ok", cost, grad )`, `( "stop", )` ends it. `threads`: the size
    of loom's CPU pool in this worker ( workers on the host share its cores )"""
    import os
    os.environ[ "CUDA_VISIBLE_DEVICES" ] = str( card )
    if threads:
        os.environ[ "SDOT_NB_THREADS" ] = os.environ[ "LOOM_NB_THREADS" ] = str( threads )
    os.environ[ "XLA_PYTHON_CLIENT_PREALLOCATE" ] = "false"
    import traceback
    import numpy as np
    try:
        import jax
        jax.config.update( "jax_enable_x64", True )
        from sdot.pixel_transport import PixelTransport2d
        u_min, v_min, du, dv = detector
        views = [ PixelTransport2d( v, [ u_min, v_min ], ( du, dv ), **setting ) for v in values ]
        conn.send( ( "ready", str( jax.devices()[ 0 ] ) ) )
    except Exception:
        conn.send( ( "error", traceback.format_exc() ) )
        return
    while True:
        msg = conn.recv()
        if msg[ 0 ] == "stop":
            return
        try:
            _, x, radius = msg
            cost, grad = 0.0, np.zeros( x.shape )
            for B, o, pt in zip( bases, offsets, views ):
                c, ( g_uv, _ ) = pt.value_and_grad( x @ B.T + o, radius )
                cost += c
                grad += np.asarray( g_uv ) @ B
            conn.send( ( "ok", cost, grad, sum( pt.nb_cold_retries for pt in views ) ) )
        except Exception:
            conn.send( ( "error", traceback.format_exc() ) )


class PixelWorkers:
    """`nb_cards` worker processes, the views dealt in turn ( view k to card k mod nb_cards ). On the host ( `LOOM_DEVICE=cpu`,
    inherited by the workers ), `threads` threads each: one view's solve keeps only a few cores busy ( 6.5 of 36 measured,
    split 4 ), so the views are spread over processes"""

    def __init__( self, nb_cards, values, bases, offsets, detector, setting, threads = None ):
        import multiprocessing as mp
        import numpy as np
        ctx = mp.get_context( "spawn" )
        self.conns, self.procs = [ ], [ ]
        for r in range( nb_cards ):
            ks = list( range( r, len( values ), nb_cards ) )
            parent, child = ctx.Pipe()
            p = ctx.Process( target = serve, daemon = True,
                             args = ( child, r, [ np.asarray( values[ k ] ) for k in ks ], [ np.asarray( bases[ k ] ) for k in ks ],
                                      [ np.asarray( offsets[ k ] ) for k in ks ], detector, setting, threads ) )
            p.start()
            self.conns.append( parent )
            self.procs.append( p )
        self.cards = [ self._answer( c )[ 1 ] for c in self.conns ]

    @staticmethod
    def _answer( conn ):
        msg = conn.recv()
        if msg[ 0 ] == "error":
            raise RuntimeError( "pixel worker:\n" + msg[ 1 ] )
        return msg

    def value_and_grad( self, x, radius ):
        for c in self.conns:
            c.send( ( "eval", x, radius ) )
        cost, grad, self.nb_cold_retries = 0.0, 0.0, 0
        for c in self.conns:
            _, ck, gk, rk = self._answer( c )
            cost, grad, self.nb_cold_retries = cost + ck, grad + gk, self.nb_cold_retries + rk
        return cost, grad

    def close( self ):
        for c, p in zip( self.conns, self.procs ):
            try:
                c.send( ( "stop", ) )
            except Exception:
                pass
            p.join( timeout = 10 )
