# Reproducing the benchmarks

All commands run from the repository root in an environment with the `gpu`, `cpu` and `viz`
extras installed. Results are written as one JSON file per case; a case whose file exists is
skipped, so interrupted runs can be restarted.

## Data

| Data | Source | Licence |
|---|---|---|
| Bentheimer sandstone, 500-cube, 5 um | Digital Rocks Portal DRP-29, doi 10.17612/P7BC78 (`python benchmarks/fetch_bentheimer.py`) | ODC-BY 1.0 |
| Pipes, five rocks and a sphere pack, 1024-cube | Saxena et al. (2017), Mendeley Data, doi 10.17632/4g723tr5v3.2 | CC BY 4.0 |

In the Saxena images label 0 is pore and 1 is solid; in the Bentheimer image 0 is pore and 255 is grain.

## Verification ladder

```bash
python benchmarks/verification.py --output results/verification \
    --groups channel slit spheres pipes wagner --saxena /path/to/saxena2017
python benchmarks/plot_verification.py --verification results/verification \
    --rock results/bentheimer384 --output results/summary
```

Groups: plane channel (tau and resolution sweeps), inclined slit (tensor orientation), simple
cubic sphere arrays against Zick and Homsy (1982), straight pipes against analytical mean
velocities, and the micromodel cell of Wagner et al. (2021). The sphere arrays hold the Reynolds
number fixed across resolutions by scaling the force with (32/n)^3.

## Rock images

```bash
# Bentheimer crop, BGK and TRT over a range of tau
python benchmarks/trt_rock.py --dataset Seg_Oxyz_0001_0001_0001.raw --crop 58 442 \
    --output results/bentheimer384 \
    --cases '[{"name":"trt_tau0.6_x","axis":0,"tau":0.6,"collision":"trt","backend":"cuda-sparse"}]'

# a 1024-cube benchmark image, float32 storage
python benchmarks/trt_rock.py --dataset Rock3_1024cube_2.072um.raw --shape 1024 1024 1024 \
    --solid-value 1 --output results/saxena \
    --cases '[{"name":"rock3_x","axis":0,"tau":0.6,"collision":"trt","backend":"cuda-sparse","precision":"float32","steps":600000,"timeout":21600}]'
```

`benchmarks/boundary_study.py` compares the periodic wrap of a rock image with mirrored domains
and with a twofold voxel refinement.

## Throughput

```bash
python benchmarks/sparse_speed.py --dataset Seg_Oxyz_0001_0001_0001.raw \
    --backend cuda-sparse --collision trt --output results/speed/cuda-sparse.json
NUMBA_NUM_THREADS=32 python benchmarks/sparse_speed.py --dataset Seg_Oxyz_0001_0001_0001.raw \
    --backend numba-sparse --collision trt --steps 100 --output results/speed/numba-32.json
```

Each file reports updates per second over all voxels and over fluid nodes only.
