"""The TRANSPORT of one view: W2^2 between the diracs of the detector's grid and the image of the projected shapes.

    w2_1d          1D detector: the closed form ( monotone plan ), differentiated by PyTorch as it stands
    grid_solver    2D detector: the calls to sdot ( dedicated grid solver, Kantorovich potential )
    GridView       2D detector: one view on it -- warm start, continuation in the blur, derivative"""
from .GridView import GridView as GridView
from .w2_1d import w2_diracs_vs_bins as w2_diracs_vs_bins
