from dataclasses import dataclass, field


@dataclass
class Machine:
    """A machine that runs otrec in the apptainer image of `torch.def` ( the environment `<name>-sif` of `errand-sif.py` )."""

    run_host: str                                                    # as in ~/.ssh/config
    work_dir: str                                                    # the copies of otrec, loom and sdot-ffi, the image, loom's cache
    layers_before_image: list = field( default_factory = list )      # e.g. `errand.Module`, `errand.Slurm`
    extra_dirs_seen_in_image: list = field( default_factory = list ) # `work_dir` and $HOME already are

    # building the image: on `run_host`, or on `image_build_host` ( with internet ) for a machine without, then copied
    image_build_host: str | None = None
    image_build_dir: str | None = None
    image_build_fakeroot: bool = True
    image_build_scratch_dir: str | None = None                       # several GB ( None: /tmp )

    # sites with an apptainer of their own ( Jean Zay )
    apptainer_exe: str = "apptainer"                                 # e.g. "singularity"
    image_install_cmd: str | None = None                             # run once the image is copied ( `{image}`: its path )
    image_installed_path: str | None = None                          # the image as run, once installed
