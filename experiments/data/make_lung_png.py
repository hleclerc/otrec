"""The clean BINARY lung of `lung_clean_2048.png` ( 2048^2, the square of side 2, the field a disc of radius 0.9 ),
made from a real slice of a lung at high resolution ( `lung_slice.webp`, noisy ):

    big structures   kept as they are segmented: solid lumps ( low texture, bright ) and airways / vessels ( low
                     texture, dark, grown a few pixels into what is dark: their edges and narrow arms read as
                     texture; the dark ones on the rim dropped as artefacts ); the bright rim of an airway and the thin
                     bands between a lumen and a lump go to the one their intensity is closer to; nothing in an airway
    parenchyma       a FOAM drawn on the whole disc, then cut by the big structures ( no layer of cells that follows
                     their borders ): Voronoi cells of random seeds relaxed a little ( Lloyd, `LLOYD` times: sizes stay
                     uneven ), the spacing of the alveoli of the slice ( its dark local minima ); the alveoli ROUNDED
                     ( the distance to the walls a smooth min of the gaps to the two nearest seeds, over `ROUND` spacing:
                     between balls and cells, wide triple junctions ); walls of a thickness that varies slowly ( a
                     smooth random field, +- `WOBBLE`, times the density of the parenchyma of the slice around ),
                     scaled so that the parenchyma keeps its mean density
    lines            the long thin bright lines found ( ridges of the Hessian at two scales, kept by hysteresis ), and
                     those drawn by hand ( `GUIDES`: the fissure between two zones, thin vessels, faint and broken in
                     the slice: a rough guide, a smooth curve fitted to the ridges near it ); no cell sliver along them
    retouches        `GUIDES`, and small airways placed by hand
    border           the disc closed by a wall

    micromamba run -n prj-torch python otrec/experiments/data/make_lung_png.py [ --preview DIR ]      ( numpy, scipy, pillow )"""
import os, sys

import numpy as np
from PIL import Image
from scipy import ndimage as nd
from scipy.spatial import cKDTree

LLOYD, WOBBLE, ROUND, N, RHO = 4, 0.25, 0.12, 2048, 0.9
here = os.path.dirname( os.path.abspath( __file__ ) )
preview = sys.argv[ sys.argv.index( "--preview" ) + 1 ] if "--preview" in sys.argv else None
im = np.asarray( Image.open( os.path.join( here, "lung_slice.webp" ) ).convert( "L" ) ).astype( float )


def keep_big( b, a ):
    lab, n = nd.label( b )
    k = np.zeros( n + 1, bool ); k[ 1: ] = nd.sum( b, lab, range( 1, n + 1 ) ) >= a
    return k[ lab ]


def local_std( x, w ):
    return np.sqrt( np.clip( nd.uniform_filter( x * x, w ) - nd.uniform_filter( x, w ) ** 2, 0, None ) )


# -- the field: the disc where the image has texture ( outside: a flat grey ) -------------------------------------------
busy = nd.binary_fill_holes( nd.binary_closing( local_std( im, 5 ) > 1.5, iterations = 3 ) )
lab, n = nd.label( busy ); disk = lab == np.argmax( nd.sum( busy, lab, range( 1, n + 1 ) ) ) + 1
cy, cx = nd.center_of_mass( disk ); r_in = np.sqrt( disk.sum() / np.pi ) - 35          # ( the rim: artefacts )
Y, X = np.mgrid[ :im.shape[ 0 ], :im.shape[ 1 ] ]; R = np.hypot( Y - cy, X - cx ); inside = R < r_in

# -- the levels of white and black ( 2-means on the flat zones ), drifting slowly -----------------------------------------
g = nd.gaussian_filter( im, 0.7 )
s15 = local_std( g, 15 )
flat = inside & ( s15 < np.percentile( s15[ inside ], 25 ) )
v = g[ flat ]; W, B = v.max(), v.min()
for _ in range( 30 ):
    t = 0.5 * ( W + B ); W, B = np.median( v[ v > t ] ), np.median( v[ v < t ] )
drift = lambda sel, lvl: nd.gaussian_filter( np.where( sel, g - lvl, 0 ), 80 ) / np.maximum( nd.gaussian_filter( sel * 1., 80 ), 1e-3 )
Wm, Bm = W + drift( flat & ( g > t ), W ), B + drift( flat & ( g < t ), B )
dens = lambda x: ( x - Bm ) / ( Wm - Bm )
f2 = dens( nd.gaussian_filter( im, 2.0 ) )

# -- big structures ------------------------------------------------------------------------------------------------------
s9 = nd.gaussian_filter( local_std( g, 7 ), 2 )
sf, st = np.median( s9[ flat ] ), np.median( s9[ inside & ~flat ] )
calm = s9 < 0.5 * ( sf + st )
lumen = nd.binary_fill_holes( keep_big( nd.binary_opening( ( f2 < 0.35 ) & calm, iterations = 2 ) & inside, 120 ) )
lab, _ = nd.label( lumen ); rim = np.unique( lab[ ( R > r_in - 12 ) & lumen ] ); lumen &= ~np.isin( lab, rim[ rim > 0 ] )
# grown ( 8 pixels at most ) into what is dark and not too textured ( the edges and the narrow arms of a lumen read as texture )
weak = ( f2 < 0.42 ) & ( s9 < 0.65 * ( sf + st ) ) & inside
lumen = nd.binary_dilation( lumen, iterations = 8, mask = weak | lumen )             # ( a few pixels at most: no leak )
lumen = nd.binary_fill_holes( nd.binary_opening( lumen, iterations = 1 ) )
# and the small ones: dark, calm, round enough to hold a disc of 4 pixels
small = nd.binary_opening( ( f2 < 0.3 ) & calm & inside, iterations = 4 ) & ~lumen
lumen |= keep_big( small, 50 )
# and those drawn by hand ( a retouch ): ( row, column, radius ) of the slice, grown into what is dark
for r0, c0, rr in [ ( 702, 237, 7 ) ]:
    lumen |= nd.binary_dilation( np.hypot( Y - r0, X - c0 ) < rr, iterations = 3, mask = weak )
solid = keep_big( nd.binary_opening( ( f2 > 0.7 ) & calm, iterations = 2 ) & inside, 25 )
solid = nd.binary_closing( solid, iterations = 2 ) & ~lumen
par = inside & ~lumen & ~solid
# thin bands of the parenchyma ( what an opening of the size of an alveolus takes off ) next to a lumen or a lump: to the
# one their intensity is closer to
gs = nd.gaussian_filter( im, 1.0 )                                                   # the alveoli: dark local minima
mins = ( gs == nd.minimum_filter( gs, 5 ) ) & ( gs < nd.uniform_filter( gs, 15 ) ) & nd.binary_erosion( par, iterations = 4 )
spacing = np.sqrt( nd.binary_erosion( par, iterations = 4 ).sum() / mins.sum() )
thin = par & ~nd.binary_opening( par, iterations = int( spacing / 2 ) ) & nd.binary_dilation( lumen | solid, iterations = 3 )
solid |= thin & ( f2 > 0.5 ); lumen |= thin & ( f2 <= 0.5 ); par &= ~thin
# the rim of an airway ( its wall, bright, whose edge reads as texture ): to the lumps, and what is left between them too thin
band = par & ( nd.distance_transform_edt( ~lumen ) < 10 ) & ( f2 > 0.55 )
solid |= band; par &= ~band
thin = par & ~nd.binary_opening( par, iterations = int( 0.7 * spacing ) ) & nd.binary_dilation( lumen | solid, iterations = 3 )
solid |= thin & ( f2 > 0.5 ); lumen |= thin & ( f2 <= 0.5 ); par &= ~thin
solid = nd.binary_closing( solid, iterations = 2 ) & ~lumen; par &= ~solid
target = float( np.clip( dens( nd.gaussian_filter( g, 3 ) )[ par ], 0, 1 ).mean() )

# -- thin bright lines: ridges at two scales, hysteresis ------------------------------------------------------------------
def ridge_strength( sig ):
    hxx, hyy, hxy = ( nd.gaussian_filter( im, sig, order = o ) for o in ( ( 2, 0 ), ( 0, 2 ), ( 1, 1 ) ) )
    root = np.sqrt( 0.25 * ( hxx - hyy ) ** 2 + hxy ** 2 )
    lmin, lmax = 0.5 * ( hxx + hyy ) - root, 0.5 * ( hxx + hyy ) + root
    return np.where( np.abs( lmax ) < 0.5 * np.abs( lmin ), -lmin * sig * sig, 0 )
rs = np.maximum( ridge_strength( 1.2 ), ridge_strength( 2.0 ) )
zone = par & ~nd.binary_dilation( lumen | solid, iterations = 8 )
strong, weak = rs > np.percentile( rs[ zone ], 98.5 ), rs > np.percentile( rs[ zone ], 90 )
lab, _ = nd.label( weak & zone )
ridge = np.isin( lab, np.unique( lab[ strong & zone ] ) ) & ( lab > 0 )
lab, n = nd.label( ridge )                                                            # long and thin: a line, not a blob
long = np.zeros( n + 1, bool )
for k, sl in enumerate( nd.find_objects( lab ), 1 ):
    yy, xx = np.nonzero( lab[ sl ] == k )
    if len( yy ) >= 100:
        ev = np.linalg.eigvalsh( np.cov( np.stack( [ yy, xx ] ) ) )
        long[ k ] = ev[ 1 ] > 25 * max( ev[ 0 ], 0.25 )
ridge = long[ lab ]
# LINES drawn by hand ( a retouch ): the fissure between two zones, and thin vessels, faint and broken in the slice. Each
# has a rough guide drawn by eye ( ( row, column ) of the slice ), the weak ridges within `window` of it, and a smooth curve
# fitted to them ( row and column as polynomials of the arc length of the guide ); drawn with `width` times the walls
GUIDES = dict(
    fissure = ( [ ( 343, 338 ), ( 350, 460 ), ( 360, 560 ), ( 380, 630 ), ( 410, 690 ), ( 450, 740 ), ( 500, 775 ), ( 560, 805 ), ( 620, 825 ), ( 660, 832 ) ], 14, 7, 1.2 ),
    vessel_upper = ( [ ( 240, 512 ), ( 290, 510 ), ( 335, 505 ) ], 8, 2, 1.5 ),
    vessel_left = ( [ ( 700, 178 ), ( 745, 180 ), ( 790, 183 ) ], 8, 2, 1.5 ) )
cand = np.stack( np.nonzero( ( rs > np.percentile( rs[ par ], 80 ) ) & inside ), 1 ).astype( float )
lines_ = { }
for name, ( guide, window, degree, width ) in GUIDES.items():
    guide = np.array( guide, float )
    seg = np.concatenate( [ np.linspace( guide[ k ], guide[ k + 1 ], 200, endpoint = False ) for k in range( len( guide ) - 1 ) ] + [ guide[ -1: ] ] )
    arc = np.concatenate( [ [ 0 ], np.cumsum( np.linalg.norm( np.diff( seg, axis = 0 ), axis = 1 ) ) ] ) / max( np.linalg.norm( np.diff( seg, axis = 0 ), axis = 1 ).sum(), 1e-9 )
    dd, jj = cKDTree( seg ).query( cand )
    near = dd < window
    sa = arc[ jj[ near ] ]
    fit = [ np.polynomial.Polynomial.fit( sa, cand[ near, c ], degree ) for c in ( 0, 1 ) ]
    sf_ = np.linspace( sa.min(), sa.max(), 4000 )
    lines_[ name ] = ( np.stack( [ fit[ 0 ]( sf_ ), fit[ 1 ]( sf_ ) ], 1 ), width )
    print( f"{ name }: { near.sum() } ridge pixels near the guide, residual { np.median( np.linalg.norm( cand[ near ] - np.stack( [ fit[ c ]( sa ) for c in ( 0, 1 ) ], 1 ), axis = 1 ) ):.2f} px" )
print( f"levels W { W:.0f} B { B:.0f}; solid { solid[ inside ].mean():.3f} lumen { lumen[ inside ].mean():.3f} parenchyma { par[ inside ].mean():.3f}"
       f" ( density { target:.3f}, { mins.sum() } alveoli, spacing { spacing:.1f} px ); ridges { ridge.sum() } px" )

# -- the output grid: the square of side 2, the disc of the field of radius RHO ------------------------------------------
sc = r_in / ( RHO * N / 2 )
i = ( np.arange( N ) - N / 2 + 0.5 ) * sc
GI, GJ = np.meshgrid( i, i, indexing = "ij" )
def up( mask, blur = 1.0 ):
    return nd.map_coordinates( nd.gaussian_filter( mask * 1., blur ), [ cy + GI, cx + GJ ], order = 1 ) > 0.5
disc = np.hypot( GI, GJ ) < r_in
L, S, P, Rg = up( lumen ), up( solid ), up( par ), up( ridge, 0.7 )
L = nd.binary_fill_holes( L )
def drawn( curve ):
    """the curve on the output grid"""
    m = np.zeros( ( N, N ), bool )
    o = np.round( ( curve - [ cy, cx ] ) / sc + N / 2 - 0.5 ).astype( int )
    o = o[ ( o >= 0 ).all( 1 ) & ( o < N ).all( 1 ) ]
    m[ o[ :, 0 ], o[ :, 1 ] ] = True
    return m
Lines = { name: ( drawn( curve ), width ) for name, ( curve, width ) in lines_.items() }
Fi = np.any( [ m for m, _ in Lines.values() ], axis = 0 )                           # every line drawn by hand
P &= disc & ~L & ~S
a = spacing / sc                                                                     # the spacing, output pixels

# -- the foam on the whole disc ------------------------------------------------------------------------------------------
rng = np.random.default_rng( 0 )
pi, pj = np.nonzero( disc ); pts = np.stack( [ pi, pj ], 1 ).astype( float )
seeds = pts[ rng.choice( len( pts ), int( len( pts ) / ( a * a ) ), replace = False ) ]
dl = nd.distance_transform_edt( ~Fi )
seeds = seeds[ dl[ tuple( np.round( seeds ).astype( int ).T ) ] > 0.4 * a ]        # no sliver of a cell along a line
for _ in range( LLOYD ):
    lab = cKDTree( seeds ).query( pts )[ 1 ]
    cnt = np.bincount( lab, minlength = len( seeds ) ); k = cnt > 0
    seeds = np.stack( [ np.bincount( lab, pts[ :, 0 ], len( seeds ) ), np.bincount( lab, pts[ :, 1 ], len( seeds ) ) ], 1 )[ k ] / cnt[ k, None ]
d, _ = cKDTree( seeds ).query( pts, 3 )
k_ = ROUND * a                                                                       # the two nearest walls, a smooth min: wider junctions
gap = -k_ * np.logaddexp( -( d[ :, 1 ] - d[ :, 0 ] ) / k_, -( d[ :, 2 ] - d[ :, 0 ] ) / k_ )
half = np.zeros( ( N, N ) ); half[ pi, pj ] = gap / 2                                # ~ distance to the walls, less at the junctions
wob = nd.gaussian_filter( rng.normal( size = ( N, N ) ), a )
wob = 1 + WOBBLE * wob / wob[ disc ].std()                                           # the thickness, varying slowly
# and following the density of the parenchyma of the slice ( smoothed over a few alveoli ): denser zones, thicker walls
fl = dens( nd.gaussian_filter( g, 3 ) ).clip( 0, 1 )
fl = nd.gaussian_filter( np.where( par, fl, 0 ), 1.5 * spacing ) / np.maximum( nd.gaussian_filter( par * 1., 1.5 * spacing ), 1e-3 )
fl = nd.map_coordinates( fl, [ cy + GI, cx + GJ ], order = 1 )
wob = wob * np.clip( fl / fl[ P ].mean(), 0.6, 1.5 )                              # ( bounded: no wall vanishes )
def foam( t ):
    """white where the walls are: walls of thickness `t wob`, the alveoli rounded"""
    return half <= 0.5 * t * wob
lo, hi = 0.0, a
for _ in range( 18 ):
    t = 0.5 * ( lo + hi )
    lo, hi = ( t, hi ) if foam( t )[ P ].mean() < target else ( lo, t )
F = foam( t )
print( f"{ len( seeds ) } cells, spacing { a:.1f} px out, walls { t:.2f} px out ( { t * sc:.2f} native ) +- { 100 * WOBBLE:.0f} %,"
       f" parenchyma density { F[ P ].mean():.3f} ( target { target:.3f} )" )
img = F & P
img |= S
img |= ( nd.distance_transform_edt( ~L ) <= 1.5 * t ) & disc & ~L                 # the walls of the airways
img |= ( nd.distance_transform_edt( ~Rg ) <= 0.5 * t ) & P                         # vessels
for m, width in Lines.values():                                                     # the lines drawn by hand
    img |= ( nd.distance_transform_edt( ~m ) <= 0.5 * width * t ) & P
img |= disc & ( np.hypot( GI, GJ ) > r_in - t * sc )                                 # the border closed
img &= disc
img &= ~L                                                                            # nothing in an airway
lab, n = nd.label( img )                                                             # nor islands nearly surrounded by one
sizes = nd.sum( img, lab, range( 1, n + 1 ) )
k = np.zeros( n + 1, bool )
for j, sl in enumerate( nd.find_objects( lab ), 1 ):
    if sizes[ j - 1 ] < 4 * a * a:
        sl2 = tuple( slice( max( x.start - 3, 0 ), x.stop + 3 ) for x in sl )
        comp = lab[ sl2 ] == j
        rim = nd.binary_dilation( comp, iterations = 2 ) & ~comp
        k[ j ] = ( rim & L[ sl2 ] ).sum() > 0.6 * rim.sum()
img &= ~k[ lab ]
print( f"islands in the airways taken off: { k.sum() }" )
print( f"white fraction in the disc { img[ disc ].mean():.3f}" )
Image.fromarray( ( img * 255 ).astype( np.uint8 ) ).save( os.path.join( here, "lung_clean_2048.png" ) )

if preview:                                                                          # the slice and the clean lung side by side
    os.makedirs( preview, exist_ok = True )
    small = np.asarray( Image.fromarray( ( img * 255 ).astype( np.uint8 ) ).resize( ( 903, 903 ), Image.BILINEAR ) ) / 255
    rgb = np.stack( [ im / 255 ] * 3, -1 ); rgb[ ridge ] = [ 1, 0.2, 0.2 ]
    for curve, _ in lines_.values():
        fi = np.clip( np.round( curve ).astype( int ), 0, np.array( im.shape ) - 1 ); rgb[ fi[ :, 0 ], fi[ :, 1 ] ] = [ 0.2, 1, 0.2 ]
    Image.fromarray( ( rgb * 255 ).astype( np.uint8 ) ).save( os.path.join( preview, "ridges.png" ) )
    for name, ( r0, c0, hw ) in dict( zoom_a = ( 480, 600, 120 ), zoom_b = ( 350, 470, 120 ), zoom_c = ( 690, 460, 120 ) ).items():
        oi, oj, ow = int( ( r0 - cy ) / sc + N / 2 ), int( ( c0 - cx ) / sc + N / 2 ), int( hw / sc )
        cl = np.asarray( Image.fromarray( ( img[ oi - ow:oi + ow, oj - ow:oj + ow ] * 255 ).astype( np.uint8 ) ).resize( ( 2 * hw, 2 * hw ), Image.BILINEAR ) ) / 255
        both = np.concatenate( [ im[ r0 - hw:r0 + hw, c0 - hw:c0 + hw ] / 255, cl ], 1 )
        Image.fromarray( ( both * 255 ).astype( np.uint8 ) ).resize( ( 960, 480 ) ).save( os.path.join( preview, name + ".png" ) )
    Image.fromarray( ( small * 255 ).astype( np.uint8 ) ).save( os.path.join( preview, "lung_903.png" ) )
