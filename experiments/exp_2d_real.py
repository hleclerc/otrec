"""MEASURED sinograms of BINARY objects ( matter or void ), from the simplest to the hardest. The data are downloaded on
first use ( `otrec.download`: into $OTREC_DATA or ~/.cache/otrec ). The page shows the shapes over the outline of the
usual reconstruction ( filtered back projection ), for reference.

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_real.py::foam'                   ( from ~/Projects )

    foam        a metal foam, Diamond I12, 2009: static, entirely in the field. The counts of the camera, with the defects
                of a real beam and detector: dark current, uneven flat, drift, beam hardening, rings, noise
    pencil      ESRF ID19, 2010: in the field, but NOT binary ( wood, glue, a graphite lead ): what the balls do with
                densities they cannot have
    sandstone   quartz grains and pores, SLS TOMCAT, 2013: the counts of the camera too, but the rock is bigger than the
                field ( matter that only some views see )
    bamboo      its fibres, ESRF ID19, 2020, 360 degrees: bigger than the field, and the rings of a detector"""
from errand import experiment, Param


def solve( rec, p ):
    """THE STRATEGY: balls at random, as small as a pixel, their centres fitted ( small, they spread as the matter does,
    whatever its shape ); then a stage at the radius where they fill the matter, and, `p.nb_splits` times, each ball
    split in 4 and the centres fitted again. Binary: the radius from the distance between neighbours ( the first peak
    of their distances: `nominal_radius` )"""
    rec.add_random_balls( p.n // 4 ** p.nb_splits, radius = rec.sinogram.h )
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )
    for split in range( p.nb_splits + 1 ):
        rec.radius = rec.nominal_radius()
        if split:
            rec.split_items( division = 4, noise_ratio = 0.5 )
        rec.adjust_parameters( centers = True, x_tol = p.x_tol )


def recorder( sino, path, title, every ):
    """the steps, the shapes coloured on demand by what the views ask of them, over the filtered back projection ( a
    layer ) and its outline: a scene store ( `path`, `xyz.sviz` ), written frame by frame while the run goes on, read by
    `sviz open path` ( `sviz export path --html page.html`: one self-contained page of it )"""
    from otrec import Recorder, viz
    fbp = sino.fbp( 512 )
    return Recorder( every = every, verbose = True, store = path,
                     page = dict( title = title, images = { "FBP": ( fbp, sino.extent ) }, truth_name = "FBP outline",
                                  truth_edges = viz.contours( fbp, sino.extent, 0.3 * fbp.max() ) ) )


# ( the sinograms are the data as they are: a bin per pixel of the detector, a view per view )
PARAMS = dict(
    n = Param( 5000, help = "shapes at the end" ),
    nb_splits = Param( 0 ),
    reduction = Param( 1, help = "the sinogram that many times coarser in pixels and in views: 1, the data as they are; 4, to try things fast" ),
    x_tol = Param( 0.1, help = "a stage stops when the shapes move less than that fraction of their radius" )
)


if p := experiment( "foam", row = Param( 500, help = "of the detector: the slice" ),
                    **dict( PARAMS, n = Param( 10000, help = "shapes at the end" ), nb_splits = Param( 1 ) ) ):
    import numpy as np
    from otrec import Reconstruction, Scan, raw_rows

    # the data: the row `p.row` of every image ( raw uint16, 1964 x 1288 ), by ranges of bytes ( the projections weigh
    # 12 GB ). Zenodo 23623238, CC BY 4.0: 10 darks, 10 flats, 1200 projections, 10 flats, 1201 projections, 10 flats
    url = "https://zenodo.org/records/23623238/files/"
    size = lambda n: ( n, 1288, 1964 )
    darks = raw_rows( url + "dark_1964_1288_10.raw", size( 10 ), p.row )
    flats = np.concatenate( [ raw_rows( url + f"flat_{ w }_1964_1288_10.raw", size( 10 ), p.row ) for w in ( "start", "middle", "end" ) ] )
    projections = raw_rows( url + "proj_1964_1288_2401.raw", size( 2401 ), p.row )

    # the steps ( the flats drift by 2 % over the scan: `minus_air` takes it off, a flat interpolated in time adds nothing )
    scan = Scan.from_counts( projections, flats, darks, angles = np.linspace( 0, np.pi, 2401 ) )
    scan = scan.rings_removed()                                       # the bias of each pixel ( 25 times the noise )
    scan = scan.minus_air( 0.03 )
    scan = scan.linearized()                                          # beam hardening: the masses vary by 4.4 % before
    scan = scan.zeroed_air( 3.0, 0.03 )                               # the air at 0, not at its clipped noise
    scan = scan.aligned()                                             # ( after `linearized`: the centres need linear data )
    scan = scan.centred( radius = 800 )                               # ( the foam is within 760 pixels of the axis )
    scan = scan.equal_masses()                                        # all in the field: the same mass in every view
    sino = scan.sinogram( reduction = p.reduction )

    rec = Reconstruction( sino, recorder = recorder( sino, p.out_dir / "foam.sviz", f"otrec -- a metal foam ( Diamond I12 ), row { p.row }", 10 ) )
    solve( rec, p )
    # rec.remove_in_air()                                               # the balls a view sees in empty air
    # rec.adjust_parameters( centers = True, x_tol = p.x_tol )
    # BINARY: a radius each, then ellipses -- the mass follows the area, one density for all
    rec.adjust_parameters( centers = True, radii = True, x_tol = p.x_tol / 10 )
    # rec.adjust_parameters( centers = True, cov_matrices = True, x_tol = p.x_tol )
    # at the END, the balls alone in the air ( no neighbour within 3 radii ): they fit the noise of the data, and a stage
    # after would put others there. ( The derivative of the loss by the mass of a ball does not tell them: they ask for MORE )
    rec.remove_isolated( 1.5 )
    rec.recorder.close( html = p.out_dir / "foam.html" )


if p := experiment( "pencil", **PARAMS ):
    import numpy as np
    from otrec import Reconstruction, Scan, cached_download

    # the data: a test sinogram of nabu ( ESRF's reconstruction software ), a pencil at 63 keV, ID19, 2010: flat-fielded,
    # -logged, 1500 views over 180 degrees, the axis given
    z = np.load( cached_download( "http://www.silx.org/pub/nabu/data/sino_pencil.npz",
                                  sha256 = "1d26a6e5006cfec07c7367faaff1e216e5349cac2ec90275c196674cdddca648" ) )

    # the steps ( the axis given, 1024.28, is 0.9 pixel off: `aligned` finds it )
    scan = Scan( z[ "data" ], angles = np.linspace( 0, np.pi, 1500, endpoint = False ) )
    scan = scan.rings_removed()                                       # the bias of each pixel ( 20 times the noise )
    scan = scan.minus_air( 0.05 )                                     # ( the air drifts by 10 % of the mass of a view )
    scan = scan.zeroed_air( 3.0, 0.05 )                               # the air at 0, not at its clipped noise
    scan = scan.aligned()                                             # the axis, and the sample moving by 1 pixel rms
    scan = scan.centred()
    scan = scan.linearized()                                          # beam hardening
    scan = scan.equal_masses()                                        # all in the field: the same mass in every view
    sino = scan.sinogram( reduction = p.reduction )

    rec = Reconstruction( sino, recorder = recorder( sino, p.out_dir / "pencil.sviz", "otrec -- a pencil ( ESRF ID19 ): wood and graphite, NOT binary", 10 ) )
    solve( rec, p )
    # the balls in the air ( a view sees them in empty air: the air is at 0 ): taken off, the others settle again. They
    # were not stuck: as many come back, beside the coating ( the densest part ), until the densities are free
    rec.remove_in_air()
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )
    # NOT binary: then a radius and a density each ( wood and graphite ), then ellipses
    rec.adjust_parameters( centers = True, radii = True, densities = True, x_tol = p.x_tol )
    rec.adjust_parameters( centers = True, cov_matrices = True, densities = True, x_tol = p.x_tol )
    rec.recorder.close( html = p.out_dir / "pencil.html" )


if p := experiment( "sandstone", slice = Param( 540, help = "270, 540, 810 or 1080" ), **PARAMS ):
    import zipfile
    import numpy as np
    import scipy.io
    from otrec import Reconstruction, Scan, cached_download

    # the data: Zenodo 4912435 ( CC BY 4.0 ), sample BBii, the 4 slices of small.zip ( 207 MB ): counts [ pixel, image ],
    # 1500 projections over 180 degrees, 0.65 um pixels, 21.5 keV
    archive = cached_download( "https://zenodo.org/records/4912435/files/small.zip", name = "sandstone_BBii_small.zip" )
    with zipfile.ZipFile( archive ) as z, z.open( f"slice_{ p.slice:04d}_data.mat" ) as f:
        m = scipy.io.loadmat( f )

    # the steps
    scan = Scan.from_counts( m[ "X_proj" ].T, m[ "X_flat" ].T, m[ "X_dark" ].T, angles = np.radians( 0.12 ) * np.arange( 1500 ) )
    scan = scan.centred( scan.rotation_axis() )
    sino = scan.clipped().sinogram( reduction = p.reduction )                 # ( no air: the rock fills the field )

    rec = Reconstruction( sino, recorder = recorder( sino, p.out_dir / "sandstone.sviz", f"otrec -- sandstone ( SLS TOMCAT ), slice { p.slice }", 100 ) )
    solve( rec, p )
    rec.recorder.close( html = p.out_dir / "sandstone.html" )


if p := experiment( "bamboo", **PARAMS ):
    import numpy as np
    from otrec import Reconstruction, Scan, cached_download

    # the data: a test sinogram of nabu ( ESRF's reconstruction software ): flat-fielded, -logged, 2000 views over 360
    # degrees, the axis given. `sino_dering`: the same, the rings filtered
    z = np.load( cached_download( "http://www.silx.org/pub/nabu/data/bamboo_rings.npz",
                                  sha256 = "3ddd21e8ce3138604163493ddea6e7b499960383d95d90b751896ca8b9ab9ef8" ) )

    # the steps
    scan = Scan( z[ "sino" ], angles = np.linspace( 0, 2 * np.pi, 2000, endpoint = False ) )
    scan = scan.centred( float( z[ "rot_center" ] ) )
    sino = scan.clipped().sinogram( reduction = p.reduction )

    rec = Reconstruction( sino, recorder = recorder( sino, p.out_dir / "bamboo.sviz", "otrec -- bamboo ( ESRF ID19 )", 100 ) )
    solve( rec, p )
    rec.recorder.close( html = p.out_dir / "bamboo.html" )
