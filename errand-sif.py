"""otrec in an apptainer image ( `torch.def` ), on the machines of `sif_machines.py`: one environment `<name>-sif` each.

`sif_machines.py` is each user's, not versioned: copy `sif_machines.example.py`. The projects are not in the image:
they are copied to `work_dir`, on PYTHONPATH.

    errand setup --env lmo-sif                    # builds the image, again after each change of torch.def
    errand run --env lmo-sif 'tests/test_loss.py::*'
"""
import errand

from .sif_machines import MACHINES


for name, m in MACHINES.items():
    errand.envs[ f"{ name }-sif" ] = errand.Env( [
        errand.Ssh(
            host = m.run_host,
            root = f"{ m.work_dir }/otrec",
            also = [ "../loom", "../sdot-ffi" ]
        ),
        *m.layers_before_image,
        errand.Apptainer(
            image = "torch.sif", recipe = "torch.def",
            flags = [ "--nv", *( f for d in [ m.work_dir, *m.extra_dirs_seen_in_image ] for f in ( "--bind", d ) ) ],
            build_on = m.image_build_host and errand.Ssh( host = m.image_build_host, root = m.image_build_dir ),
            fakeroot = m.image_build_fakeroot,
            scratch = m.image_build_scratch_dir,
            exe = m.apptainer_exe,
            install = m.image_install_cmd,
            run_from = m.image_installed_path
        ),
        errand.Vars( {
            "PYTHONPATH"    : f"{ m.work_dir }/otrec/src:{ m.work_dir }/loom/src:{ m.work_dir }/sdot-ffi/src",
            "LOOM_CACHE_DIR": f"{ m.work_dir }/.cache/loom",     # not in $HOME, which is small on a cluster ( 3 GB on Jean Zay )
            "OMPI_MCA_pml"  : "ob1",
            "OMPI_MCA_mtl"  : "^psm2,ofi",
            "OMPI_MCA_btl"  : "self,vader,tcp",
        } ),
    ], needs = { "gpus": 1 },       # every run puts its tensors on the card ( loom's default device )
       driver = "torch", cuda = True )

# errand.Env( [
#     errand.Ssh( host = "jz", root = "/lustre/fswork/projects/rech/vst/urg11uj/<projet>" ),
#     errand.Apptainer( image = "containers/mpi.sif", recipe = "containers/mpi.def",
#                       fakeroot = True, scratch = "/data/apptainer-scratch",
#                       build_on = errand.Ssh( host = "lmo", root = "/home/leclerc/<projet>" ),
#                       install = 'bash -lc "module load singularity && idrcontmgr cp {image}"',
#                       run_from = "$SINGULARITY_ALLOWED_DIR/mpi.sif", exe = "singularity",
#                       flags = [ "--bind", "/lustre/fswork" ] ),
# ] )
#