"""Read-only resource snapshot and real CUDA allocation/kernel test."""
import json, os, platform, subprocess, sys, time
from pathlib import Path

def command(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def snapshot():
    d = {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'host': platform.node(), 'platform': platform.platform(), 'python': sys.version,
         'executable': sys.executable, 'thread_settings': {k: os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
         'commands': {name: command(args) for name,args in {
             'cpu':['lscpu'], 'memory':['free','-b'], 'disk':['df','-B1','.'],
             'gpu':['nvidia-smi'], 'load':['uptime'],
             'processes':['ps','-eo','pid,comm,pcpu,pmem','--sort=-pcpu'],
             'dependencies':[sys.executable,'-m','pip','freeze'],
             'commit':['git','rev-parse','HEAD'], 'status':['git','status','--porcelain=v1']}.items()}}
    try:
        import cupy as cp
        x = cp.arange(256, dtype=cp.float64)
        kernel=cp.RawKernel('extern "C" __global__ void twice(double* x) {int i=threadIdx.x; x[i]*=2.;}', 'twice')
        kernel((1,), (256,), (x,))
        cp.cuda.runtime.deviceSynchronize()
        assert float(x.sum()) == 65280
        d['cuda_test'] = {'passed': True,'cupy':cp.__version__, 'runtime':cp.cuda.runtime.runtimeGetVersion(),
                          'driver':cp.cuda.runtime.driverGetVersion(),'free_total_bytes':cp.cuda.runtime.memGetInfo()}
    except Exception as e:
        d['cuda_test'] = {'passed':False,'error':repr(e)}
    return d

if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps(snapshot(), indent=2))
