import math

import numpy as np


def gaussian_matrix( n, sigma ):
    """`[ n, n ]`: the gaussian blur of width `sigma` ( in samples ) of `n` regularly spaced samples ( identity for 0 ),
    numpy"""
    if sigma <= 0:
        return np.eye( n )
    i = np.arange( n )
    g = np.exp( -0.5 * ( ( i[ :, None ] - i[ None, : ] ) / sigma ) ** 2 )
    return g / ( math.sqrt( 2 * math.pi ) * sigma )
