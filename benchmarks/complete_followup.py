"""Sequential authorized follow-up after the 256³ tolerance check passes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
root=Path('results/2026-09-18-pilot')
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
check=root/'convergence_review/tolerance_256.json'
while not check.exists():
 try:os.kill(a.wait_pid,0)
 except ProcessLookupError:raise RuntimeError('convergence process exited without its summary')
 time.sleep(5)
if not json.loads(check.read_text())['passed']:
 raise RuntimeError('tighter-tolerance comparison failed; investigate before tensor campaign')
# Wait for the producing process to release its CUDA context as well.
while True:
 try:os.kill(a.wait_pid,0)
 except ProcessLookupError:break
 time.sleep(1)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name,args in [
 ('solid_shortcut_probe',['benchmarks/probe_solid_shortcut.py']),
 ('end_to_end',['benchmarks/end_to_end.py']),
 ('long_rock',['benchmarks/long_rock.py','--output',str(root/'long_rock'),
  '--reuse-x',str(root/'convergence_review/rock_256_standard/result.json')])]:
 status=dict(phase=name,state='running',timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 (root/'followup_status.json').write_text(json.dumps(status,indent=2)+'\n')
 with (root/(name+'.log')).open('x') as log:
  result=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
 status.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
 (root/'followup_status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(name,status['state'],flush=True)
 if result.returncode:raise RuntimeError(name+' failed; see saved log')
