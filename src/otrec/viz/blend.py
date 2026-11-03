"""A Blender scene ( `.blend` ) of a 3D reconstruction run, next to its pages: see `blend_scene.py`, which Blender runs.

    python -m otrec.viz.blend runs/.../run.npz            # scene.blend next to it ( and the pages again, with the
                                                          # object's surface when it is a Voronoi foam )

Blender is looked for in `$BLENDER`, then on the PATH, then in /Applications ( macOS ). Only the scene is built here,
nothing is rendered: open it, press F12 ( or Ctrl-F12 for the animation ).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def blender_path():
    for c in ( os.environ.get( "BLENDER" ), shutil.which( "blender" ), "/Applications/Blender.app/Contents/MacOS/Blender" ):
        if c and Path( c ).exists():
            return c
    return None


def truth_voxels( run_dir ):
    """the real object as voxels ( its solid fraction, those its radiographs were made from ), from the run's
    `summary.json`: `voxels.npz` next to it, or None when the phantom is not a Voronoi foam"""
    import json
    import numpy as np
    from ..phantoms import VoronoiFoam
    run_dir = Path( run_dir )
    p = json.loads( ( run_dir / "summary.json" ).read_text() )[ "params" ]
    if p.get( "phantom" ) != "voronoi":
        return None
    sh = p.get( "seed_height" ) or p[ "height" ]
    nb_cells = max( 8, int( round( np.pi * 0.8 ** 2 * sh / p[ "cell" ] ** 3 ) ) )
    ph = VoronoiFoam( nb_cells = nb_cells, porosity = p[ "porosity" ], height = p[ "height" ], seed = p.get( "seed", 0 ), seed_height = sh )
    vol, h, origin = ph.voxels( p[ "pixels" ] )
    out = run_dir / "voxels.npz"
    np.savez_compressed( out, vol = vol, h = h, origin = origin )
    return out


def write_blend( npz, out = None ):
    """`scene.blend` next to `npz` ( or `out` ); None when there is no Blender. For a Voronoi foam the real object is a
    surface ( the isosurface 0.5 of its voxels ), also written as `surface.npz` for the pages ( `view3d` )"""
    exe = blender_path()
    if exe is None:
        return None
    npz = Path( npz ).resolve()
    out = Path( out ) if out else npz.with_name( "scene.blend" )
    script = Path( __file__ ).with_name( "blend_scene.py" )
    vox = truth_voxels( npz.parent ) if ( npz.parent / "summary.json" ).exists() else None
    r = subprocess.run( [ exe, "-b", "--factory-startup", "-P", str( script ), "--", str( npz ), str( out.resolve() ),
                          str( vox ) if vox else "" ], capture_output = True, text = True )
    if r.returncode != 0 or not out.exists():
        raise RuntimeError( f"blender failed:\n{ r.stdout[ -3000: ] }\n{ r.stderr[ -3000: ] }" )
    print( [ l for l in r.stdout.splitlines() if l.startswith( "wrote" ) ][ -1 ] )
    return out


if __name__ == "__main__":
    out = write_blend( sys.argv[ 1 ] )
    print( out )
    if out is not None and Path( sys.argv[ 1 ] ).with_name( "surface.npz" ).exists():   # the pages, with the surface
        from .view3d import write_view
        print( write_view( sys.argv[ 1 ] ) )
