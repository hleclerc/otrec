"""Run on the host even where there is a card, and even when jax and loom are already imported ( as in an errand entry,
where setting `JAX_PLATFORMS` / `LOOM_DEVICE` comes too late )."""
import os


def use_cpu():
    os.environ[ "LOOM_DEVICE" ] = "cpu"
    os.environ[ "JAX_PLATFORMS" ] = "cpu"
    import jax
    try:
        jax.config.update( "jax_platforms", "cpu" )     # effective only before the backends are initialized
    except Exception:
        pass
    jax.config.update( "jax_default_device", jax.devices( "cpu" )[ 0 ] )
    import loom
    loom.default_device = "cpu"
