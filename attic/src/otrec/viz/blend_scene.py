"""Runs INSIDE Blender ( `blender -b -P blend_scene.py -- run.npz out.blend` ): a scene of a 3D reconstruction run.

- the object first ( a sample of it, gray ), then one point cloud per snapshot, each shown in turn ( `HOLD` frames
  each ), then a turntable around the last one;
- the diracs are POINTS of a geometry-nodes cloud, rendered as spheres by Cycles, their radius half the snapshot's
  spacing ( neighbours touch; fewer diracs, each carrying more mass, are bigger ), colored by a blue -> white -> red
  ramp of an attribute: `mass` ( the mass derivative, symmetric around 0: blue the dirac would rather grow, red
  shrink ) or `pull` ( the disagreement of the views ), both scaled PER SNAPSHOT ( its own quantiles ); the material's
  `Show` value picks it ( 0 pull, 1 mass -- the default, 2 plain );
- a floor just under the object, that its walls cast their shadows on ( the depth cue ), a cylinder's wireframe when
  the phantom has one, three area lights ( key, fill, rim ), a dim world, Cycles with denoising and the AgX view.

With a third argument ( `voxels.npz`: the real object's solid fraction ), the object is not a sample but its SURFACE:
the voxels become an OpenVDB grid ( Blender bundles `openvdb` ), a Volume object, and geometry nodes take its
isosurface 0.5 ( Volume to Mesh ), smooth-shaded; that mesh is also written as `surface.npz` ( vertices, triangles )
for the HTML pages.

Blender's own Python and numpy: nothing from otrec is imported here.
"""
import math
import sys

import bpy
import numpy as np

HOLD = 12          # frames per snapshot
TURN = 180         # frames of the final turntable


def frames( d ):
    sizes = d[ "f_sizes" ] if "f_sizes" in d.files else np.full( len( d[ "f_step" ] ), d[ "f_points" ].shape[ 1 ] )
    flat, cut = d[ "f_points" ].reshape( -1, 3 ), np.concatenate( [ [ 0 ], np.cumsum( sizes ) ] )
    pull, mass = d[ "f_disagreement" ].reshape( -1 ), d[ "f_mass_grad" ].reshape( -1 )
    return [ ( int( d[ "f_step" ][ f ] ), flat[ cut[ f ]:cut[ f + 1 ] ], pull[ cut[ f ]:cut[ f + 1 ] ], mass[ cut[ f ]:cut[ f + 1 ] ] )
             for f in range( len( sizes ) ) ]


def scales( pull, mass ):
    """a snapshot's scale: the pull's quantiles, the 98th percentile of | mass derivative |"""
    return np.quantile( pull, np.linspace( 0, 1, 101 ) ), float( np.quantile( np.abs( mass ), 0.98 ) or 1.0 )


def normalized( pull, mass, sc ):
    """pull by its quantiles ( 0 .. 1 ), mass symmetric around 0.5 ( saturated at its 98th percentile )"""
    q, m = sc
    return np.interp( pull, q, np.linspace( 0, 1, len( q ) ) ), np.clip( 0.5 + 0.5 * mass / m, 0, 1 )


def spacing( x ):
    """the median distance to the nearest neighbour ( on a sample, by brute force in blocks: no scipy in Blender )"""
    rng = np.random.default_rng( 0 )
    s = x[ rng.choice( len( x ), min( 2000, len( x ) ), replace = False ) ]
    best = np.full( len( s ), np.inf )
    for i in range( 0, len( x ), 20000 ):
        d = ( ( s[ :, None, : ] - x[ None, i:i + 20000 ] ) ** 2 ).sum( 2 )
        d[ d == 0 ] = np.inf
        best = np.minimum( best, d.min( 1 ) )
    return float( np.median( np.sqrt( best ) ) )


def material():
    mat = bpy.data.materials.new( "diracs" )
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new( "ShaderNodeOutputMaterial" )
    bsdf = nt.nodes.new( "ShaderNodeBsdfPrincipled" )
    bsdf.inputs[ "Roughness" ].default_value = 0.45
    show = nt.nodes.new( "ShaderNodeValue" )
    show.name = show.label = "Show"
    show.outputs[ 0 ].default_value = 1.0
    a_pull, a_mass = nt.nodes.new( "ShaderNodeAttribute" ), nt.nodes.new( "ShaderNodeAttribute" )
    a_pull.attribute_name, a_mass.attribute_name = "pull", "mass"
    ramp = nt.nodes.new( "ShaderNodeValToRGB" )
    els = ramp.color_ramp.elements
    els[ 0 ].color = ( 0.03, 0.17, 0.55, 1 )
    els[ 1 ].position, els[ 1 ].color = 1.0, ( 0.62, 0.02, 0.05, 1 )
    mid = els.new( 0.5 )
    mid.color = ( 0.92, 0.92, 0.92, 1 )
    # Show: 0 -> pull, 1 -> mass ( a mix ), 2 -> plain gray
    mix = nt.nodes.new( "ShaderNodeMix" )
    mix.data_type = "FLOAT"
    clamp1 = nt.nodes.new( "ShaderNodeClamp" )
    nt.links.new( show.outputs[ 0 ], clamp1.inputs[ "Value" ] )
    nt.links.new( clamp1.outputs[ 0 ], mix.inputs[ "Factor" ] )
    nt.links.new( a_pull.outputs[ "Fac" ], mix.inputs[ 2 ] )
    nt.links.new( a_mass.outputs[ "Fac" ], mix.inputs[ 3 ] )
    nt.links.new( mix.outputs[ 0 ], ramp.inputs[ "Fac" ] )
    gray = nt.nodes.new( "ShaderNodeMix" )
    gray.data_type = "RGBA"
    sub = nt.nodes.new( "ShaderNodeMath" )
    sub.operation = "SUBTRACT"
    nt.links.new( show.outputs[ 0 ], sub.inputs[ 0 ] )
    sub.inputs[ 1 ].default_value = 1.0
    clamp2 = nt.nodes.new( "ShaderNodeClamp" )
    nt.links.new( sub.outputs[ 0 ], clamp2.inputs[ "Value" ] )
    nt.links.new( clamp2.outputs[ 0 ], gray.inputs[ "Factor" ] )
    nt.links.new( ramp.outputs[ "Color" ], gray.inputs[ 6 ] )
    gray.inputs[ 7 ].default_value = ( 0.75, 0.75, 0.75, 1 )
    nt.links.new( gray.outputs[ 2 ], bsdf.inputs[ "Base Color" ] )
    nt.links.new( bsdf.outputs[ 0 ], out.inputs[ "Surface" ] )
    return mat


def truth_material():
    mat = bpy.data.materials.new( "object" )
    mat.use_nodes = True
    b = mat.node_tree.nodes[ "Principled BSDF" ]
    b.inputs[ "Base Color" ].default_value = ( 0.62, 0.62, 0.60, 1 )
    b.inputs[ "Roughness" ].default_value = 0.5
    return mat


def points_group( radius, mat ):
    """geometry nodes: the mesh's vertices -> points of `radius`, with the material"""
    ng = bpy.data.node_groups.new( f"points r={ radius:.4g}", "GeometryNodeTree" )
    ng.interface.new_socket( "Geometry", in_out = "INPUT", socket_type = "NodeSocketGeometry" )
    ng.interface.new_socket( "Geometry", in_out = "OUTPUT", socket_type = "NodeSocketGeometry" )
    gi, go = ng.nodes.new( "NodeGroupInput" ), ng.nodes.new( "NodeGroupOutput" )
    m2p = ng.nodes.new( "GeometryNodeMeshToPoints" )
    m2p.inputs[ "Radius" ].default_value = radius
    sm = ng.nodes.new( "GeometryNodeSetMaterial" )
    sm.inputs[ "Material" ].default_value = mat
    ng.links.new( gi.outputs[ 0 ], m2p.inputs[ "Mesh" ] )
    ng.links.new( m2p.outputs[ "Points" ], sm.inputs[ "Geometry" ] )
    ng.links.new( sm.outputs[ "Geometry" ], go.inputs[ 0 ] )
    return ng


def cloud( name, x, attrs, group, coll ):
    me = bpy.data.meshes.new( name )
    me.vertices.add( len( x ) )
    me.vertices.foreach_set( "co", np.asarray( x, dtype = np.float32 ).ravel() )
    for k, v in attrs.items():
        a = me.attributes.new( k, "FLOAT", "POINT" )
        a.data.foreach_set( "value", np.asarray( v, dtype = np.float32 ) )
    ob = bpy.data.objects.new( name, me )
    coll.objects.link( ob )
    mod = ob.modifiers.new( "points", "NODES" )
    mod.node_group = group
    return ob


def surface( vox_path, coll, mat ):
    """the isosurface 0.5 of the voxels: an OpenVDB grid, a Volume object, Volume to Mesh"""
    import os
    import openvdb as vdb
    v = np.load( vox_path )
    vol, h, o = v[ "vol" ].astype( np.float32 ), float( v[ "h" ] ), v[ "origin" ]
    grid = vdb.FloatGrid()
    grid.copyFromArray( np.ascontiguousarray( np.transpose( vol, ( 1, 2, 0 ) ) ) )   # [ z, x, y ] -> ( i, j, k ) = ( x, y, z )
    grid.transform = vdb.createLinearTransform( [ [ h, 0, 0, 0 ], [ 0, h, 0, 0 ], [ 0, 0, h, 0 ], [ o[ 0 ], o[ 1 ], o[ 2 ], 1 ] ] )
    grid.name = "density"
    path = os.path.join( os.path.dirname( vox_path ), "object.vdb" )
    vdb.write( path, grids = [ grid ] )
    vd = bpy.data.volumes.new( "object volume" )
    vd.filepath = path
    vob = bpy.data.objects.new( "object volume", vd )
    coll.objects.link( vob )
    vob.hide_viewport = vob.hide_render = True
    # a MESH object whose geometry nodes read the volume ( Object Info ) and take its isosurface: a mesh object can be
    # evaluated to a mesh ( for the pages ), a volume object cannot
    ob = bpy.data.objects.new( "object", bpy.data.meshes.new( "object" ) )
    coll.objects.link( ob )
    ng = bpy.data.node_groups.new( "surface", "GeometryNodeTree" )
    ng.interface.new_socket( "Geometry", in_out = "INPUT", socket_type = "NodeSocketGeometry" )
    ng.interface.new_socket( "Geometry", in_out = "OUTPUT", socket_type = "NodeSocketGeometry" )
    go = ng.nodes.new( "NodeGroupOutput" )
    info = ng.nodes.new( "GeometryNodeObjectInfo" )
    info.inputs[ "Object" ].default_value = vob
    info.transform_space = "RELATIVE"
    v2m = ng.nodes.new( "GeometryNodeVolumeToMesh" )
    v2m.inputs[ "Threshold" ].default_value = 0.5
    smooth = ng.nodes.new( "GeometryNodeSetShadeSmooth" )
    sm = ng.nodes.new( "GeometryNodeSetMaterial" )
    sm.inputs[ "Material" ].default_value = mat
    ng.links.new( info.outputs[ "Geometry" ], v2m.inputs[ "Volume" ] )
    ng.links.new( v2m.outputs[ "Mesh" ], smooth.inputs[ "Geometry" ] )
    ng.links.new( smooth.outputs[ "Geometry" ], sm.inputs[ "Geometry" ] )
    ng.links.new( sm.outputs[ "Geometry" ], go.inputs[ 0 ] )
    mod = ob.modifiers.new( "surface", "NODES" )
    mod.node_group = ng
    bpy.context.view_layer.update()
    # the mesh, for the pages: lighter ( voxels twice as big, flat parts simplified ), then the full one back for Blender
    v2m.inputs[ "Resolution Mode" ].default_value = "Size"
    v2m.inputs[ "Voxel Size" ].default_value = 3 * h
    v2m.inputs[ "Adaptivity" ].default_value = 0.9
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get( dg ).to_mesh()
    me.calc_loop_triangles()
    verts = np.empty( 3 * len( me.vertices ), np.float32 )
    me.vertices.foreach_get( "co", verts )
    tris = np.empty( 3 * len( me.loop_triangles ), np.int32 )
    me.loop_triangles.foreach_get( "vertices", tris )
    np.savez_compressed( os.path.join( os.path.dirname( vox_path ), "surface.npz" ), vertices = verts.reshape( -1, 3 ), triangles = tris.reshape( -1, 3 ) )
    print( f"surface for the pages: { len( me.vertices ) } vertices, { len( me.loop_triangles ) } triangles" )
    ob.evaluated_get( dg ).to_mesh_clear()
    v2m.inputs[ "Resolution Mode" ].default_value = "Grid"
    v2m.inputs[ "Adaptivity" ].default_value = 0.0
    return ob


def show_between( ob, f0, f1 ):
    """visible ( viewport and render ) on frames [ f0, f1 ), hidden elsewhere"""
    for f, hide in ( ( 0, True ), ( f0, False ), ( f1, True ) ):
        if f < 0:
            continue
        ob.hide_viewport = ob.hide_render = hide
        ob.keyframe_insert( "hide_viewport", frame = f )
        ob.keyframe_insert( "hide_render", frame = f )


def wire_cylinder( c, coll ):
    x, y, z, r, h = c[ :5 ]
    bpy.ops.mesh.primitive_cylinder_add( vertices = 96, radius = r, depth = h, location = ( x, y, z ) )
    ob = bpy.context.active_object
    for cl in ob.users_collection:
        cl.objects.unlink( ob )
    coll.objects.link( ob )
    ob.name = "cylinder"
    w = ob.modifiers.new( "wire", "WIREFRAME" )
    w.thickness = 0.002 * r
    mat = bpy.data.materials.new( "wire" )
    mat.use_nodes = True
    mat.node_tree.nodes[ "Principled BSDF" ].inputs[ "Base Color" ].default_value = ( 0.3, 0.3, 0.3, 1 )
    ob.data.materials.append( mat )
    return ob


def area_light( name, loc, energy, size, color, target ):
    ld = bpy.data.lights.new( name, "AREA" )
    ld.energy, ld.size, ld.color = energy, size, color
    ob = bpy.data.objects.new( name, ld )
    bpy.context.scene.collection.objects.link( ob )
    ob.location = loc
    tc = ob.constraints.new( "TRACK_TO" )
    tc.target, tc.track_axis, tc.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    return ob


def main( npz, out, vox = "" ):
    d = np.load( npz )
    bpy.ops.wm.read_factory_settings( use_empty = True )
    sc = bpy.context.scene
    coll = bpy.data.collections.new( "reconstruction" )
    sc.collection.children.link( coll )

    snaps = frames( d )
    xl = snaps[ -1 ][ 1 ]
    lo, hi = xl.min( 0 ), xl.max( 0 )
    center, size = ( lo + hi ) / 2, float( np.max( hi - lo ) )
    # the radius: half each snapshot's spacing, neighbours touch ( the same for every snapshot of a stage )
    mat = material()
    groups = { }

    def group_for( x, m ):
        n = len( x )
        if ( n, m.name ) not in groups:
            groups[ ( n, m.name ) ] = points_group( 0.5 * spacing( x ), m )
        return groups[ ( n, m.name ) ]

    ts = d[ "truth_sample" ]
    obs = [ surface( vox, coll, truth_material() ) if vox else cloud( "object", ts, { }, group_for( ts, truth_material() ), coll ) ]
    for step, x, pull, mass in snaps:
        p, m = normalized( pull, mass, scales( pull, mass ) )
        obs.append( cloud( f"step {step:05d}", x, dict( pull = p, mass = m ), group_for( x, mat ), coll ) )
    rad = 0.5 * spacing( xl )
    for i, ob in enumerate( obs ):
        last = i == len( obs ) - 1
        show_between( ob, i * HOLD + 1, ( i + 1 ) * HOLD + 1 + ( TURN if last else 0 ) )
    for c in d[ "cylinders" ] if "cylinders" in d.files else [ ]:
        wire_cylinder( c, coll )

    # the camera on a pivot: a slow turn along the snapshots, a full one at the end
    pivot = bpy.data.objects.new( "pivot", None )
    sc.collection.objects.link( pivot )
    pivot.location = center
    cam_d = bpy.data.cameras.new( "camera" )
    cam_d.lens = 50
    cam = bpy.data.objects.new( "camera", cam_d )
    sc.collection.objects.link( cam )
    cam.parent = pivot
    dist = 1.6 * size
    el = math.radians( 50 )
    cam.location = ( 0, -dist * math.cos( el ), dist * math.sin( el ) )
    tc = cam.constraints.new( "TRACK_TO" )
    tc.target, tc.track_axis, tc.up_axis = pivot, "TRACK_NEGATIVE_Z", "UP_Y"
    sc.camera = cam
    end = len( obs ) * HOLD + TURN
    pivot.rotation_euler = ( 0, 0, 0 )
    pivot.keyframe_insert( "rotation_euler", frame = 1 )
    pivot.rotation_euler = ( 0, 0, math.radians( 60 ) )
    pivot.keyframe_insert( "rotation_euler", frame = len( obs ) * HOLD )
    pivot.rotation_euler = ( 0, 0, math.radians( 420 ) )
    pivot.keyframe_insert( "rotation_euler", frame = end )

    # a FLOOR just under the object: the walls cast their shadows on it, seen through the cells
    s = size
    bpy.ops.mesh.primitive_plane_add( size = 8 * s, location = ( center[ 0 ], center[ 1 ], lo[ 2 ] - 0.04 * s ) )
    floor = bpy.context.active_object
    floor.name = "floor"
    fm = bpy.data.materials.new( "floor" )
    fm.use_nodes = True
    fb = fm.node_tree.nodes[ "Principled BSDF" ]
    fb.inputs[ "Base Color" ].default_value = ( 0.045, 0.046, 0.05, 1 )
    fb.inputs[ "Roughness" ].default_value = 0.6
    floor.data.materials.append( fm )

    # light: key ( warm, high, small: hard shadows ), fill ( cool, low, wide ), rim ( behind, above )
    area_light( "key", center + np.array( [ -1.2 * s, -0.9 * s, 2.4 * s ] ), 70 * s * s, 0.25 * s, ( 1.0, 0.94, 0.86 ), pivot )
    area_light( "fill", center + np.array( [ 1.8 * s, -1.0 * s, 0.5 * s ] ), 10 * s * s, 1.5 * s, ( 0.80, 0.87, 1.0 ), pivot )
    area_light( "rim", center + np.array( [ 0.3 * s, 1.9 * s, 1.2 * s ] ), 45 * s * s, 0.8 * s, ( 1.0, 1.0, 1.0 ), pivot )
    world = bpy.data.worlds.new( "world" )
    world.use_nodes = True
    bg = world.node_tree.nodes[ "Background" ]
    bg.inputs[ "Color" ].default_value = ( 0.02, 0.022, 0.026, 1 )
    bg.inputs[ "Strength" ].default_value = 1.0
    sc.world = world

    sc.render.engine = "CYCLES"
    sc.view_settings.exposure = -0.3
    sc.cycles.samples = 96
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 1600, 1200
    sc.render.fps = 24
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.frame_start, sc.frame_end = 1, end
    sc.frame_set( len( obs ) * HOLD )
    bpy.ops.wm.save_as_mainfile( filepath = out )
    print( f"wrote { out }: { len( obs ) } clouds, frames 1..{ end }, point radius { rad:.4g}" )


if __name__ == "__main__":
    argv = sys.argv[ sys.argv.index( "--" ) + 1: ]
    main( argv[ 0 ], argv[ 1 ], argv[ 2 ] if len( argv ) > 2 else "" )
