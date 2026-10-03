"""The common palette for reconstruction figures.

The hues for CURVES come from Okabe-Ito, which is safe for deuteranopia and protanopia -- the figures
are read in grayscale in print as often as on screen. MAPS follow the
usual rule: a single light->dark hue for a positive quantity (density, sinogram),
a diverging one with a neutral midpoint for a SIGNED quantity (a residual).
"""

#: Okabe-Ito: blue, vermilion, bluish green, plus a grey for reference marks
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

#: sequential (positive quantities) and diverging (signed quantities)
SEQ, DIV = "magma_r", "RdBu_r"
