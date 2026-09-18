"""Read-only resource sampling; exits on a stop-file or the supplied time limit."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=3600);a=p.parse_args()
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);stop=path.with_suffix('.stop')
start=time.perf_counter()
with path.open('x') as f:
 while time.perf_counter()-start<a.seconds and not stop.exists():
  mem={}
  for line in Path('/proc/meminfo').read_text().splitlines():
   key,rest=line.split(':',1)
   if key in ('MemTotal','MemAvailable'):mem[key+'_bytes']=int(rest.split()[0])*1024
  q=subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
  row=dict(elapsed_s=time.perf_counter()-start,load_average=os.getloadavg(),host=mem,
   gpu_csv=q.stdout.strip(),gpu_fields=['timestamp','memory_used_MiB','memory_free_MiB','gpu_util_percent','memory_util_percent','power_W'],
   note='device-wide samples include concurrent processes; not exact per-case peaks')
  f.write(json.dumps(row)+'\n');f.flush();time.sleep(5)
