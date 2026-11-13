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
        } ),
    ], driver = "torch", cuda = True )
