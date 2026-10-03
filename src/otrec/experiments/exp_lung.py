"""Lung alveoli reconstruction experiment.

Usage:
    ./run experiment lung
    ./run experiment lung --nb-diracs=5000 --max-iter=100
"""
from errand import Param, experiment

if p := experiment(
    "lung BFGS",
    nb_diracs=Param(10_000, help="Number of Dirac masses"),
    max_iter=Param(60, help="Max number of LBFGS iterations"),
):
    from otrec.experiments.lung_alveoli import run
    run(nb_diracs=p.nb_diracs, max_iter=p.max_iter)
