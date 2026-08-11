"""Point d'entrée UNIQUE de la reconstruction CT : la classe `Reconstruction`.

Elle porte l'ÉTAT (le sinogramme mesuré, le nuage de points courant) et les PARAMÈTRES PAR
DÉFAUT (rayon des disques, finesse de la grille modèle, optimiseur, graine, enregistrement des
frames), et expose les algorithmes comme des méthodes CHAÎNABLES : chacune part du nuage courant,
le remplace, et renvoie `self`.

    rec = Reconstruction( sino, radius = 0.15, record = True, verbose = True )
    rec.random_points( 60 ).diracs( max_iter = 100 ).disks( max_iter = 300 )
    rec.export_html( "out.html" )

C'est ce chaînage qui permet de COMPOSER les deux modèles de `models.py` : démarrer par une
reconstruction en DIRACS (bon marché, insensible au tirage initial, mais dont la perte n'est pas
différentiable partout) puis reprendre EXACTEMENT les mêmes points comme centres de DISQUES, dont
la perte est lisse -- chaque étape n'ayant plus qu'à raffiner ce que la précédente a trouvé.

Aucune méthode ne suppose l'ordre : `disks()` seul, `diracs()` seul, `multiscale()` puis `disks()`
sont tous valides. L'historique (`history`) garde une ligne par étape, et `frames` la trajectoire
complète quand `record` est actif -- de quoi rejouer l'ensemble de l'enchaînement dans une seule
page HTML.
"""
import time

import numpy as np

from sdot import Tensor, driver

from .Sinogram import Sinogram
from .models import DiracModel, DiskModel, Model
from .optimizers import FusedLBFGS, LBFGS
from .viz.points_html import export_positions_html


class Reconstruction:
    """Reconstruction CT d'un nuage de points à partir d'un sinogramme, par étapes chaînables.

    `sinogram` : la donnée à laquelle les points sont confrontés. Constante en pratique, mais
    remplaçable (`set_sinogram`) -- c'est ce dont a besoin l'alternance halo/intérieur de
    `halo.alternate`, qui resoumet à chaque passe un sinogramme corrigé.
    `points` : nuage initial `[ n, 2 ]` (Tensor ou tableau) -- optionnel, `random_points` peut le
    tirer plus tard.

    Paramètres par défaut, tous surchargeables à l'appel de chaque étape :
    - `radius` / `nb_pixels` / `max_chunk_elems` : le modèle DISQUES (`radius` est obligatoire pour
      `disks()`, mais peut être donné soit ici, soit à l'appel ; `max_chunk_elems` fixe le pic
      mémoire du gradient, indépendamment du nombre de disques -- voir `DiskProjector.values`) ;
    - `with_barycenters` : le modèle DIRACS (compromis mémoire/vitesse du backward) ;
    - `optimizer`, ou à défaut `max_iter`/`ftol` qui définissent le L-BFGS utilisé ;
    - `extent` : l'étendue des tirages/du bruit de split (par défaut celle du détecteur) ;
    - `seed` : la graine de TOUT tirage aléatoire (incrémentée à chaque usage, pour que deux
      tirages successifs ne soient pas identiques) ;
    - `record` / `record_every` : capture des positions au fil des pas, pour l'animation ;
    - `verbose` : une ligne de résumé par étape.
    """

    def __init__( self, sinogram: Sinogram, points = None, *,
                  radius: float | None = None, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24, with_barycenters: bool = False,
                  optimizer = None, max_iter: int = 200, ftol: float = 1e-12,
                  min_iter: int = 0, disp_tol: float | None = None,
                  extent: float | None = None, seed: int = 0,
                  record: bool = False, record_every: int = 1, verbose: bool = False ) -> None:
        self.sinogram = sinogram

        self.radius = radius
        self.nb_pixels = nb_pixels
        self.max_chunk_elems = max_chunk_elems
        self.with_barycenters = with_barycenters

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.ftol = ftol
        #: voir `LBFGS.min_iter`/`disp_tol` -- défauts appliqués à CHAQUE étage (`diracs`/`disks`/
        #: `multiscale`) qui ne fournit pas son propre `optimizer`, surchargeables à l'appel.
        self.min_iter = min_iter
        self.disp_tol = disp_tol

        self.extent = float( extent if extent is not None else sinogram.extent )
        self.seed = int( seed )
        self.record = record
        self.record_every = max( 1, int( record_every ) )
        self.verbose = verbose

        #: le nuage courant, `[ n, 2 ]` -- l'état que chaque étape transforme
        self.points: Tensor | None = None
        #: trajectoire capturée (liste de tableaux `[ n_t, 2 ]`), si `record`
        self.frames: list[ np.ndarray ] = []
        #: une ligne par étape exécutée : modèle, optimiseur, pertes, nombre de pas, durée
        self.history: list[ dict ] = []
        #: rayon monde des points, hérité du dernier modèle joué (voir `export_html`)
        self.radii = None

        if points is not None:
            self.set_points( points )

    # -- état --------------------------------------------------------------

    @property
    def nb_points( self ) -> int:
        if self.points is None:
            return 0
        return int( self.points.shape[ 0 ] )

    @property
    def positions( self ) -> np.ndarray:
        """Le nuage courant en numpy `[ n, 2 ]` (copie hôte, pour la mesure/l'affichage)."""
        return np.asarray( self.points )

    def set_points( self, points ) -> "Reconstruction":
        """Remplace le nuage courant. Accepte un Tensor, un tableau `[ n, 2 ]`, ou une autre
        `Reconstruction` (on reprend alors son nuage -- pour repartir d'un résultat existant)."""
        if isinstance( points, Reconstruction ):
            points = points.points
        raw = points.raw if isinstance( points, Tensor ) else driver.array( np.asarray( points, dtype = float ) )
        if raw.ndim != 2 or raw.shape[ 1 ] != 2:
            raise ValueError( f"points doit être de shape [ n, 2 ], reçu { tuple( raw.shape ) }" )
        self.points = Tensor.wrap( raw, [ "num_point", "dim" ] )
        return self

    def set_sinogram( self, sinogram: Sinogram ) -> "Reconstruction":
        """Remplace la donnée à laquelle les points sont confrontés, en GARDANT le nuage courant.

        Les modèles sont reconstruits à chaque étape depuis `self.sinogram` (`dirac_model` /
        `disk_model`), rien n'en est mis en cache ici : l'échange est donc immédiat. Sert à
        l'alternance halo/intérieur (`halo.alternate`), où chaque passe repart du nuage précédent
        -- réchauffé, donc -- sur un sinogramme mieux corrigé.
        """
        self.sinogram = sinogram
        return self

    def _next_seed( self, seed: int | None ) -> int:
        """La graine à utiliser : celle demandée, ou la graine par défaut -- qu'on FAIT AVANCER,
        pour que deux tirages successifs sur le même objet ne se répètent pas."""
        if seed is not None:
            return int( seed )
        self.seed += 1
        return self.seed - 1

    def random_points( self, nb_points: int, extent: float | None = None,
                       seed: int | None = None ) -> "Reconstruction":
        """`nb_points` positions 2D tirées uniformément dans [ -extent/2, extent/2 ]^2."""
        rng = np.random.default_rng( self._next_seed( seed ) )
        e = float( extent if extent is not None else self.extent )
        return self.set_points( ( rng.random( ( nb_points, 2 ) ) - 0.5 ) * e )

    def split( self, factor: int = 4, noise_frac: float = 0.05,
               seed: int | None = None ) -> "Reconstruction":
        """Remplace chaque point par `factor` enfants, superposés au parent puis dispersés d'un
        bruit uniforme (par axe).

        `noise_frac` : amplitude du bruit en fraction de l'espacement typique au niveau COURANT
        (`extent / sqrt( n )`) -- les enfants doivent explorer le voisinage IMMÉDIAT du parent,
        sans quoi chaque raffinement redeviendrait un réarrangement global (voir `multiscale`).
        """
        if self.points is None:
            raise ValueError( "aucun point à découper -- appeler `random_points` ou `set_points` d'abord" )
        rng = np.random.default_rng( self._next_seed( seed ) )
        noise_scale = noise_frac * self.extent / np.sqrt( max( 1, self.nb_points ) )
        tiled = np.repeat( self.positions, factor, axis = 0 )                     # [ n*factor, 2 ]
        return self.set_points( tiled + ( rng.random( tiled.shape ) - 0.5 ) * 2 * noise_scale )

    def subsample( self, nb_points: int, seed: int | None = None ) -> "Reconstruction":
        """Garde `nb_points` points tirés sans remise (sans effet s'il y en a déjà moins)."""
        if self.nb_points <= nb_points:
            return self
        idx = np.random.default_rng( self._next_seed( seed ) ).choice( self.nb_points, nb_points, replace = False )
        return self.set_points( self.positions[ idx ] )

    # -- modèles et optimiseur ---------------------------------------------

    def dirac_model( self, with_barycenters: bool | None = None ) -> DiracModel:
        return DiracModel( self.sinogram,
                           with_barycenters = self.with_barycenters if with_barycenters is None else with_barycenters )

    def disk_model( self, radius: float | None = None, nb_pixels: int | None = None,
                    max_chunk_elems: int | None = None ) -> DiskModel:
        r = self.radius if radius is None else radius
        if r is None:
            raise ValueError( "le modèle disques exige un rayon : `Reconstruction( ..., radius = ... )` "
                              "ou `disks( radius = ... )`" )
        return DiskModel(
            self.sinogram, radius = r,
            nb_pixels = self.nb_pixels if nb_pixels is None else nb_pixels,
            max_chunk_elems = self.max_chunk_elems if max_chunk_elems is None else max_chunk_elems,
        )

    def default_model( self ) -> Model:
        """Le modèle sous-entendu quand on n'en précise aucun (`loss()`, `multiscale()`) : DISQUES
        si un rayon a été fixé à la construction, DIRACS sinon."""
        return self.dirac_model() if self.radius is None else self.disk_model()

    def default_optimizer( self, max_iter: int | None = None, ftol: float | None = None,
                           min_iter: int | None = None, disp_tol: float | None = ...,
                           fused: bool = False ):
        """L'optimiseur explicitement fourni à la construction, ou à défaut un L-BFGS calibré par
        `max_iter`/`ftol`/`min_iter`/`disp_tol` (voir `LBFGS`, le plus efficace mesuré sur ce
        problème par ailleurs, cf. `benchmarks/optimizers`).

        `disp_tol` distingue `None` (voulu, désactive le critère) de "non fourni" (`...`, hérite
        du défaut de la `Reconstruction`) -- les trois autres paramètres n'ont pas ce besoin,
        `None` y signifiant déjà "hérite du défaut".

        `fused` : construit un `FusedLBFGS` (voir `Reconstruction.diracs( backend = "sycl" )`) au
        lieu d'un `LBFGS` -- ignore alors `self.optimizer` (un optimiseur générique fourni à la
        construction ne sait pas consommer un `value_and_grad` fusionné)."""
        if ( not fused and self.optimizer is not None and max_iter is None and ftol is None
             and min_iter is None and disp_tol is ... ):
            return self.optimizer
        cls = FusedLBFGS if fused else LBFGS
        return cls( max_iter = self.max_iter if max_iter is None else max_iter,
                   ftol = self.ftol if ftol is None else ftol,
                   min_iter = self.min_iter if min_iter is None else min_iter,
                   disp_tol = self.disp_tol if disp_tol is ... else disp_tol )

    # -- évaluation --------------------------------------------------------

    def loss( self, model: Model | None = None, points = None ) -> float:
        """Perte du nuage (courant, ou `points` s'il est fourni) pour `model` (ou `default_model`)."""
        model = model or self.default_model()
        pts = self.points if points is None else points
        if pts is None:
            raise ValueError( "aucun point à évaluer" )
        return float( model.cost( pts ) )

    def floor( self, model: Model | None = None ) -> float:
        """Le coût incompressible du modèle -- la valeur à laquelle comparer `loss()`."""
        return ( model or self.default_model() ).floor

    # -- algorithmes -------------------------------------------------------

    def run( self, model: Model, optimizer = None, max_iter: int | None = None,
             ftol: float | None = None, min_iter: int | None = None, disp_tol: float | None = ...,
             callback = None, label: str | None = None ) -> "Reconstruction":
        """UNE étape d'optimisation : descend `model.cost` depuis le nuage courant, qu'elle
        remplace par le résultat.

        `callback( step, points )` est appelé à chaque pas, `step = -1` désignant l'état initial.
        L'enregistrement des frames (si `record`) et la ligne d'`history` sont gérés ici, donc
        communs à toutes les étapes -- `diracs`/`disks`/`multiscale` ne font que choisir le modèle.

        `min_iter`/`disp_tol` : voir `LBFGS` -- ignorés si `optimizer` est fourni explicitement.
        """
        if self.points is None:
            raise ValueError( "aucun point de départ -- appeler `random_points` ou `set_points` d'abord" )
        optimizer = optimizer if optimizer is not None else self.default_optimizer( max_iter, ftol, min_iter, disp_tol )

        p = self.points.raw
        loss_before = float( model.cost( model.wrap( p ) ) )
        nb_steps = [ 0 ]

        # l'état initial n'est capturé qu'une fois : chaîner deux étapes ne doit pas dupliquer la
        # frame de jonction (le point de départ de l'une EST l'arrivée de l'autre).
        if self.record and not self.frames:
            self._capture( p )
        if callback is not None:
            callback( -1, model.wrap( p ) )

        def scalar_loss( q ):
            return model.cost( model.wrap( q ) ).tensor

        def step_callback( step, x ):
            nb_steps[ 0 ] = step + 1
            if self.record and ( step + 1 ) % self.record_every == 0:
                self._capture( x )
            if callback is not None:
                callback( step, model.wrap( x ) )

        t0 = time.time()
        # `FusedLBFGS` consomme `model.value_and_grad` (coût+gradient fusionnés, ex. le kernel
        # SYCL de `DiracModel`) au lieu d'une fonction scalaire jax-traçable -- voir
        # `optimizers.FusedLBFGS` et `Reconstruction.diracs( backend = "sycl" )`.
        if isinstance( optimizer, FusedLBFGS ):
            fused = getattr( model, "value_and_grad", None )
            if fused is None:
                raise ValueError( f"{ model !r } ne supporte pas l'évaluation fusionnée "
                                  "(FusedLBFGS) -- il lui manque `value_and_grad`" )
            p_opt = optimizer.minimize( fused, p, callback = step_callback )
        else:
            p_opt = optimizer.minimize( scalar_loss, p, callback = step_callback )
        dt = time.time() - t0

        self.set_points( model.wrap( p_opt ) )
        self.radii = model.radii
        loss_after = float( model.cost( model.wrap( self.points.raw ) ) )

        self.history.append( dict(
            stage = len( self.history ), model = model.name, label = label or model.name,
            optimizer = type( optimizer ).__name__, nb_points = self.nb_points,
            loss_before = loss_before, loss_after = loss_after, floor = model.floor,
            nb_steps = nb_steps[ 0 ], time = dt,
        ) )
        if self.verbose:
            print( self._summary_line( self.history[ -1 ] ) )
        return self

    def diracs( self, with_barycenters: bool | None = None, backend: str = "jax",
               **kwargs ) -> "Reconstruction":
        """Une étape avec le modèle DIRACS (voir `models.DiracModel`). `kwargs` -> `run`.

        `backend` : `"jax"` (défaut) descend `model.cost` via `jax.grad`, généraliste (compatible
        avec n'importe quel `optimizer`). `"sycl"` utilise à la place le kernel SYCL fusionné
        `dirac_sycl.diracs_cost_grad` (`DiracModel.value_and_grad`, coût ET gradient EN UN SEUL
        passage, formule fermée -- ~10-30x plus rapide mesuré sur le poumon, voir
        `benchmarks/execution_speed/benchmark_fused.py`) : force l'optimiseur par défaut à
        `FusedLBFGS` (voir `default_optimizer`), sauf si `optimizer` est fourni explicitement.
        """
        model = self.dirac_model( with_barycenters )
        if backend == "sycl":
            kwargs.setdefault( "optimizer", self.default_optimizer(
                kwargs.get( "max_iter" ), kwargs.get( "ftol" ),
                kwargs.get( "min_iter" ), kwargs.get( "disp_tol", ... ), fused = True ) )
        elif backend != "jax":
            raise ValueError( f"backend inconnu : { backend !r } (attendu 'jax' ou 'sycl')" )
        return self.run( model, **kwargs )

    def disks( self, radius: float | None = None, nb_pixels: int | None = None,
               max_chunk_elems: int | None = None, split_before: int = 1,
               split_noise_frac: float = 0.05, **kwargs ) -> "Reconstruction":
        """Une étape avec le modèle DISQUES (voir `models.DiskModel`). `kwargs` -> `run` (dont
        `min_iter`/`disp_tol`, voir `LBFGS` -- utiles ici : la perte disques a des directions
        PLATES -- ex. un disque peut glisser sans changer le résidu tant qu'il ne chevauche
        personne -- où L-BFGS-B peut conclure à convergence après 0-1 pas).

        Typiquement enchaînée APRÈS `diracs`/`multiscale` : les points convergés y deviennent les
        centres des disques, sans autre transformation.

        `split_before` : si > 1, SUBDIVISE le nuage courant (voir `split`) juste avant l'étage
        disques -- plus de centres à rayon fixe donnent plus de degrés de liberté pour se
        réarranger localement. Utile si le nuage hérité (typiquement un dirac par case du
        sinogramme, voir `models.sinogram_diracs`) est trop clairsemé pour le rayon demandé et se
        retrouve VERROUILLÉ (chevauchements imposés par le voisinage, aucun déplacement local ne
        peut baisser la perte). `1` (défaut) ne change rien -- comportement historique.
        """
        if split_before > 1:
            self.split( factor = split_before, noise_frac = split_noise_frac )
        return self.run( self.disk_model( radius, nb_pixels, max_chunk_elems ), **kwargs )

    def multiscale( self, nb_points_final: int, nb_points_init: int = 1000, factor: int = 4,
                    noise_frac: float = 0.05, model: Model | None = None,
                    optimizer_factory = None, stage_callback = None, **kwargs ) -> "Reconstruction":
        """Grossier -> fin : converge à `nb_points_init` points, puis répète { `split` (chaque point
        -> `factor` enfants bruités), reconverger } jusqu'à `nb_points_final`.

        Motivation : à `nb_points_final` directement (ex. 1e7), un tirage initial UNIFORME doit être
        globalement réarrangé d'un coup par l'optimiseur (la masse cible est concentrée sur une
        toute petite fraction du support) -- observé à faire échouer la line search de L-BFGS-B
        (arrêt après 1-2 itérations, loin de la convergence). Par étapes, chaque niveau hérite de la
        bonne structure GLOBALE et n'a plus qu'à raffiner LOCALEMENT : bien mieux conditionné.

        Part du nuage courant s'il y en a un (on peut donc raffiner un résultat existant), sinon
        d'un tirage uniforme de `nb_points_init` points. `optimizer_factory( n )`, s'il est fourni,
        donne l'optimiseur de l'étage à `n` points ; `stage_callback( stage, n, points )` est appelé
        après convergence de chaque étage, avant le split suivant.
        """
        model = model or self.default_model()
        if self.points is None:
            self.random_points( nb_points_init )

        stage = 0
        while True:
            optimizer = None if optimizer_factory is None else optimizer_factory( self.nb_points )
            self.run( model, optimizer = optimizer, label = f"{ model.name } x{ self.nb_points }", **kwargs )
            if stage_callback is not None:
                stage_callback( stage, self.nb_points, self.points )
            if self.nb_points >= nb_points_final:
                return self

            # `factor` peut déborder la cible au dernier étage : on sous-échantillonne alors, plutôt
            # que de résoudre un facteur non entier.
            next_n = min( self.nb_points * factor, nb_points_final )
            self.split( factor, noise_frac = noise_frac ).subsample( next_n )
            stage += 1

    # -- sorties -----------------------------------------------------------

    def _capture( self, raw ) -> None:
        # float32 : les frames ne servent qu'à l'affichage, et une trajectoire longue à grand
        # nombre de points est vite plus lourde que la reconstruction elle-même.
        self.frames.append( np.array( raw, dtype = np.float32, copy = True ) )

    def _summary_line( self, h: dict ) -> str:
        floor = f" (plancher { h[ 'floor' ]:.8f})" if h[ "floor" ] > 0 else ""
        return ( f"[{ h[ 'label' ] }] { h[ 'nb_points' ] } points, { h[ 'optimizer' ] } : "
                 f"perte { h[ 'loss_before' ]:.6f} -> { h[ 'loss_after' ]:.8f}{ floor }"
                 f"  ({ h[ 'nb_steps' ] } pas, { h[ 'time' ]:.1f}s)" )

    def summary( self ) -> str:
        """L'historique complet des étapes, une ligne chacune."""
        return "\n".join( self._summary_line( h ) for h in self.history )

    def export_html( self, out_path: str, extent: float | None = None, animate: bool | None = None,
                     radii = ..., **kwargs ) -> "Reconstruction":
        """Écrit la page HTML autonome du nuage (voir `viz.points_html.export_positions_html`).

        `animate` : rejoue la trajectoire enregistrée (défaut : dès qu'il y a des frames, donc dès
        que `record` était actif) ; `False` n'exporte que le nuage final.

        `radii` : par défaut le rayon du DERNIER modèle joué -- fixé et donc EXPORTÉ après une étape
        disques (les disques sont alors dessinés à leur taille réelle, le slider n'étant plus qu'un
        facteur d'échelle), `None` après une étape diracs (le slider EST le rayon d'affichage).
        """
        if self.points is None:
            raise ValueError( "rien à exporter -- aucun point" )
        animate = bool( self.frames ) if animate is None else animate
        if animate and not self.frames:
            raise ValueError( "aucune frame enregistrée -- construire avec `record = True`" )
        export_positions_html(
            self.frames if animate else self.positions,
            extent = float( extent if extent is not None else self.extent ),
            out_path = out_path,
            radii = self.radii if radii is ... else radii,
            **kwargs,
        )
        return self
