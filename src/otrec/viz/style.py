"""La palette commune aux figures de reconstruction.

Les teintes des COURBES viennent d'Okabe-Ito, sûre en deutéranopie et protanopie -- les figures
sont lues en niveaux de gris à l'impression aussi souvent qu'à l'écran. Les CARTES suivent la
règle usuelle : une seule teinte claire->foncée pour une grandeur positive (densité, sinogramme),
une divergente à milieu neutre pour une grandeur SIGNÉE (un résidu).
"""

#: Okabe-Ito : bleu, vermillon, vert-bleu, plus un gris pour les repères
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

#: séquentielle (grandeurs positives) et divergente (grandeurs signées)
SEQ, DIV = "magma_r", "RdBu_r"
