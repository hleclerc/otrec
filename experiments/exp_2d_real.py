"""MEASURED sinograms of BINARY objects ( matter or void ), from the simplest to the hardest. The data are downloaded on
first use ( `otrec.download`: into $OTREC_DATA or ~/.cache/otrec ). The page shows the shapes over the outline of the
usual reconstruction ( filtered back projection ), for reference.

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_real.py::foam'                   ( from ~/Projects )

    foam        a metal foam, Diamond I12, 2009: static, entirely in the field. The counts of the camera, with the defects
                of a real beam and detector: dark current, uneven flat, drift, beam hardening, rings, noise
    sandstone   quartz grains and pores, SLS TOMCAT, 2013: the counts of the camera too, but the rock is bigger than the
                field ( matter that only some views see )
    bamboo      its fibres, ESRF ID19, 2020, 360 degrees: bigger than the field, and the rings of a detector"""
from errand import experiment, Param


def solve( rec, p ):
    """THE STRATEGY: balls at random, their centres fitted; then, `p.nb_splits` times, each ball split in 4 and the
    centres fitted again. Binary: each stage at the radius where the balls, spaced as they are, fill the matter"""
    rec.add_random_balls( p.n // 4 ** p.nb_splits, radius = 0.0 )
    for split in range( p.nb_splits + 1 ):
        rec.radius = rec.nominal_radius()
        if split:
            rec.split_items( division = 4, noise_ratio = 1.0 )
        rec.adjust_parameters( centers = True, x_tol = p.x_tol )


def page( rec, sino, path, title ):
    from otrec import viz
    fbp = sino.fbp()
    rec.recorder.write_html( path, title = title, truth_edges = viz.contours( fbp, sino.extent, 0.3 * fbp.max() ) )


PARAMS = dict( n = Param( 40000, help = "shapes at the end" ), nb_splits = Param( 3 ), bins = Param( 512 ), views = Param( 800 ),
               x_tol = Param( 0.05, help = "a stage stops when the shapes move less than that fraction of their radius" ) )


if p := experiment( "foam", row = Param( 500, help = "of the detector: the slice" ), **PARAMS ):
    import numpy as np
    from otrec import Reconstruction, Recorder, Scan, raw_rows

    # the data: the row `p.row` of every image ( raw uint16, 1964 x 1288 ), by ranges of bytes ( the projections weigh
    # 12 GB ). Zenodo 23623238, CC BY 4.0: 10 darks, 10 flats, 1200 projections, 10 flats, 1201 projections, 10 flats
    url = "https://zenodo.org/records/23623238/files/"
    size = lambda n: ( n, 1288, 1964 )
    darks = raw_rows( url + "dark_1964_1288_10.raw", size( 10 ), p.row )
    flats = np.concatenate( [ raw_rows( url + f"flat_{ w }_1964_1288_10.raw", size( 10 ), p.row ) for w in ( "start", "middle", "end" ) ] )
    projections = raw_rows( url + "proj_1964_1288_2401.raw", size( 2401 ), p.row )

    # the steps
    scan = Scan.from_counts( projections, flats, darks, angles = np.linspace( 0, np.pi, 2401 ) )
    scan = scan.centred( scan.rotation_axis(), radius = 800 )          # ( the foam is within 760 pixels of the axis )
    scan = scan.minus_air( 0.03 )
    scan = scan.linearized()                                          # beam hardening
    sino = scan.clipped().sinogram( p.bins, p.views )

    rec = Reconstruction( sino, recorder = Recorder( every = 100, verbose = True ) )
    solve( rec, p )
    page( rec, sino, p.out_dir / "foam.html", f"otrec -- a metal foam ( Diamond I12 ), row { p.row }" )


if p := experiment( "sandstone", slice = Param( 540, help = "270, 540, 810 or 1080" ), **PARAMS ):
    import zipfile
    import numpy as np
    import scipy.io
    from otrec import Reconstruction, Recorder, Scan, cached_download

    # the data: Zenodo 4912435 ( CC BY 4.0 ), sample BBii, the 4 slices of small.zip ( 207 MB ): counts [ pixel, image ],
    # 1500 projections over 180 degrees, 0.65 um pixels, 21.5 keV
    archive = cached_download( "https://zenodo.org/records/4912435/files/small.zip", name = "sandstone_BBii_small.zip" )
    with zipfile.ZipFile( archive ) as z, z.open( f"slice_{ p.slice:04d}_data.mat" ) as f:
        m = scipy.io.loadmat( f )

    # the steps
    scan = Scan.from_counts( m[ "X_proj" ].T, m[ "X_flat" ].T, m[ "X_dark" ].T, angles = np.radians( 0.12 ) * np.arange( 1500 ) )
    scan = scan.centred( scan.rotation_axis() )
    sino = scan.clipped().sinogram( p.bins, p.views )                 # ( no air: the rock fills the field )

    rec = Reconstruction( sino, recorder = Recorder( every = 100, verbose = True ) )
    solve( rec, p )
    page( rec, sino, p.out_dir / "sandstone.html", f"otrec -- sandstone ( SLS TOMCAT ), slice { p.slice }" )


if p := experiment( "bamboo", **PARAMS ):
    import numpy as np
    from otrec import Reconstruction, Recorder, Scan, cached_download

    # the data: a test sinogram of nabu ( ESRF's reconstruction software ): flat-fielded, -logged, 2000 views over 360
    # degrees, the axis given. `sino_dering`: the same, the rings filtered
    z = np.load( cached_download( "http://www.silx.org/pub/nabu/data/bamboo_rings.npz",
                                  sha256 = "3ddd21e8ce3138604163493ddea6e7b499960383d95d90b751896ca8b9ab9ef8" ) )

    # the steps
    scan = Scan( z[ "sino" ], angles = np.linspace( 0, 2 * np.pi, 2000, endpoint = False ) )
    scan = scan.centred( float( z[ "rot_center" ] ) )
    sino = scan.clipped().sinogram( p.bins, p.views )

    rec = Reconstruction( sino, recorder = Recorder( every = 100, verbose = True ) )
    solve( rec, p )
    page( rec, sino, p.out_dir / "bamboo.html", "otrec -- bamboo ( ESRF ID19 )" )
