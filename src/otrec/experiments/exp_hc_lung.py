"""Lung alveoli reconstruction — `HcReconstruction` demo.

Run via:
    ./run experiment hc_lung
    ./run experiment hc_lung --nb-diracs=10000 --max-iter=60
"""
from otrec.HcReconstruction import HcReconstruction
from loom.cli import experiment, Param
import time

if p := experiment( "hc_lung",
    nb_alveoli = Param( 1000, help = "Number of alveoli (holes)" ),
    nb_diracs = Param( 10000, help = "Number of Dirac masses" ),
    nb_angles = Param( 600, help = "Nb angles" ),
    max_iter = Param( 40, help = "Max line-search iterations" ),
    backend = Param( "sycl", help = "OT backend: jax | sycl" ),
):

    hc, lobes, alveoli = HcReconstruction.make_lung_phantom( nb_alveoli = p.nb_alveoli, backend = p.backend, nb_angles = p.nb_angles, record = True )
    points = hc.random_points( p.nb_diracs // 4**2, seed = 1 )
    t0 = time.time()
    for _ in range( 3 ):
        hc.line_search( points, max_iter = p.max_iter )
        points = hc.split()
    dt = time.time() - t0
    print( dt )

    hc.export_html( "tmp/hc_lung.html" )
