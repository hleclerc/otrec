"""Copy to `sif_machines.py` ( not versioned ), with your hosts, logins and directories. Read by `errand-sif.py`."""
import errand

from .sif_machine import Machine

MACHINES = {}

# a simple workstation
MACHINES[ "ws"] = Machine( 
    run_host = "my-workstation", 
    work_dir = "/home/LOGIN/otrec-sif" 
)

# Jean Zay: no internet, so built on the workstation; only the images copied by `idrcontmgr` run there
for arch, cpus in [ ( "h100", 24 ), ( "a100", 8 ) ]:
    MACHINES[ f"jz-{ arch }" ] = Machine(
        run_host                = "jz",
        work_dir                = "/lustre/fswork/projects/rech/PROJECT/LOGIN/otrec-sif",   # in $WORK: $HOME is 3 GB
        layers_before_image     = [
            errand.Module( "singularity" ),
            errand.Slurm( 
                account = f"PROJECT@{ arch }", 
                qos = f"qos_gpu_{ arch }-dev", 
                extra = [ "-C", arch, "--hint=nomultithread" ],
                gpus = 1, cpus = cpus, time = "00:20:00", watch = 300 
            ),
        ],
        image_build_host        = "my-workstation",
        image_build_dir         = "/home/LOGIN/otrec-sif-build",
        apptainer_exe           = "singularity",
        image_install_cmd       = 'bash -lc "module load singularity && idrcontmgr cp {image}"',
        image_installed_path    = "$SINGULARITY_ALLOWED_DIR/torch.sif",
    )
