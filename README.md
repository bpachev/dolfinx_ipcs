
This is a modified version of Jørgen S. Dokken's FEniCSx IPCS solver for the Navier-Stokes equations. It adds support for running the benchmark script with GPU acceleration. The original README can be found [here](https://github.com/jorgensd/dolfinx_ipcs/blob/main/README.md).

# Installation

The code in this repository requires [GMSH](https://gmsh.info/), including the Python interface, [tqdm](https://github.com/tqdm/tqdm), and [CUDOLFINx]((https://github.com/bpachev/cuda-dolfinx)).

# [DFG 3D benchmark](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html)

The resolution of the mesh can be changed by modifying the lc variable in `cfd.geo`. Two higher-resolution geofiles are provided with the repository for convenience - `cfd.geo-lc003` and `cfd.geo-lc0006`. The original three-dimensional mesh generation script has been modified to allow generation of multiple named meshes. For example:

```bash
python3 create_and_convert_3D_mesh.py --geofile=cfd.geo-lc003 --filename=channel3D-lc003
```

To solve the problem with CUDA acceleration, run

```bash
python3 DFG_benchmark.py --cuda --filename=channel3D-lc003
```

The benchmark can be run as normal by excluding the --cuda option, which may be desireable when comparing GPU-accelerated performance to a parallel CPU baseline.
