"""Bounded, sequential, resumable rock campaign; no population checkpoints.

Usage: python benchmarks/campaign.py --config FILE --output DIRECTORY --budget 28800
The parent uses stdlib only; a single child at a time owns the CUDA context.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
GIB=1024**3


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for folder in ('lbm_permeability','benchmarks','tests')
            for p in sorted((ROOT/folder).glob('*.py'))}


def resources():
    memory={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')
    return dict(time=time.time(),host_available_bytes=memory['MemAvailable'],
        gpu_free_bytes=int(gpu[0])*1024**2,gpu_used_bytes=int(gpu[1])*1024**2,
        gpu_utilization_percent=int(gpu[2]),load_average=os.getloadavg(),
        disk_free_bytes=shutil.disk_usage(ROOT).free)


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def case_fingerprint(job):
    # Output location does not affect mathematics; all actual settings and source do.
    obj={k:v for k,v in job.items() if k not in ('output','fingerprint','gpu_pool_limit_bytes')}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def limit_cores(count=16):
    allowed=os.sched_getaffinity(0);selected=[];seen=set()
    rows=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    for line in rows.splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','))
        if cpu in allowed and (core,socket) not in seen:
            selected.append(cpu);seen.add((core,socket))
        if len(selected)==count:break
    os.sched_setaffinity(0,selected)
    return selected


class Campaign:
    def __init__(self,config,output,budget):
        self.config=config;self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.sources=source_hashes();self.started=time.time();self.deadline=self.started+budget
        self.rows=[];self.selected=None
        frozen=self.output/'frozen.json'
        identity=dict(config=config,source_sha256=self.sources)
        if frozen.exists() and json.loads(frozen.read_text())!=identity:
            raise ValueError('Resume refused: source/config changed. Use a new output directory.')
        save(frozen,identity)
        # A resumed invocation has an explicitly recorded new budget, never a hidden extension.
        self.session=self.output/('session_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
        self.session.mkdir()
        save(self.session/'start.json',dict(started=self.started,deadline=self.deadline,
             budget_s=budget,physical_core_logical_ids=limit_cores(config.get('cpu_cores',16)),resources=resources()))
        if digest(config['dataset'])!=config['dataset_sha256']:raise ValueError('Dataset checksum mismatch')
        for filename,args in [('git_status.txt',['git','status','--porcelain=v1']),
                              ('git_commit.txt',['git','rev-parse','HEAD']),
                              ('dependencies.txt',[sys.executable,'-m','pip','freeze'])]:
            (self.session/filename).write_text(subprocess.check_output(args,text=True,cwd=ROOT))
        source_dir=self.output/'source'
        for name in self.sources:
            target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(ROOT/name,target)

    def remaining(self):return self.deadline-time.time()

    def task(self,name,kind,crop,cap,**kw):
        if self.remaining()<630:return None
        if source_hashes()!=self.sources:raise RuntimeError('Source changed during campaign; refusing further tasks')
        observation=resources()
        needed=kw.pop('projected_disk_bytes',0)
        if observation['host_available_bytes']<50*GIB+8*GIB:
            raise RuntimeError('Insufficient host RAM reserve for next task')
        if directory_bytes(self.output)+needed>self.config.get('disk_budget_gib',50)*GIB:
            raise RuntimeError('Campaign disk budget would be exceeded')
        if observation['disk_free_bytes']<needed+5*GIB:
            raise RuntimeError('Insufficient disk reserve')
        job=dict(kind=kind,crop=crop,dataset=self.config['dataset'],
            dataset_sha256=self.config['dataset_sha256'],dataset_shape=self.config['dataset_shape'],
            voxel_size_m=self.config['voxel_size_m'],source_sha256=self.sources,
            gpu_pool_limit_bytes=int(.8*observation['gpu_free_bytes']),**kw)
        job['fingerprint']=case_fingerprint(job)
        base=self.output/name;base.mkdir(parents=True,exist_ok=True)
        for old in sorted(base.glob('attempt_*')):
            done=old/'complete.json'
            if done.exists() and json.loads(done.read_text())['fingerprint']==job['fingerprint']:
                result=json.loads((old/'result.json').read_text()) if (old/'result.json').exists() else {}
                # Failed/timed-out numerical results are retained but not reused as successes.
                accepted=result.get('valid_for_permeability',False)
                fixed=job.get('preflight') or job.get('fixed_steps')
                fields_ok=not job.get('settings',{}).get('return_fields') or all((old/f).exists() for f in ('ux.npy','uy.npy','uz.npy','rho.npy','fields.vti','field_metadata.json'))
                if fields_ok and (kind!='solve' or accepted or (fixed and result.get('iterations')==job['settings']['n_steps_max'])):
                    self.rows.append(dict(case=name,path=str(old),execution='reused',status=result.get('termination_reason','ok')))
                    self.inventory();return old
        attempt=base/f'attempt_{len(list(base.glob("attempt_*")))+1:02d}'
        attempt.mkdir();job['output']=str(attempt);save(attempt/'job.json',job)
        print(f'{time.strftime("%H:%M:%S")} START {name}, cap={cap:.0f}s, remaining={self.remaining():.0f}s',flush=True)
        limit=min(cap,self.remaining()-600);started=time.time();samples=[];reason='completed'
        with (attempt/'stdout.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'benchmarks/campaign_worker.py'),str(attempt/'job.json')],
                stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            try:
                while child.poll() is None:
                    sample=resources();samples.append(sample)
                    if time.time()-started>limit:reason='parent_wall_timeout'
                    elif sample['host_available_bytes']<50*GIB:reason='host_memory_reserve'
                    elif sample['disk_free_bytes']<5*GIB:reason='disk_reserve'
                    elif sample['gpu_free_bytes']<.2*observation['gpu_free_bytes']:reason='gpu_reserve'
                    if reason!='completed':
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(3)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill();child.wait()
        save(attempt/'resource_samples.json',dict(before=observation,samples=samples,
            interpretation='device-wide NVIDIA observations include other processes; host available and load are system-wide'))
        save(attempt/'execution.json',dict(reason=reason,returncode=child.returncode,elapsed_s=time.time()-started))
        r=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
        self.rows.append(dict(case=name,path=str(attempt),execution=reason,
            status=r.get('termination_reason','failed' if child.returncode else 'ok')))
        self.inventory()
        print(f'{time.strftime("%H:%M:%S")} END {name}: {self.rows[-1]["status"]}',flush=True)
        return attempt if child.returncode==0 and (attempt/'complete.json').exists() else None

    def inventory(self):
        with (self.output/'case_inventory.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['case','path','execution','status']);w.writeheader();w.writerows(self.rows)
        save(self.output/'status.json',dict(started=self.started,deadline=self.deadline,
              remaining_s=self.remaining(),selected=self.selected,cases=self.rows))

    def solve(self,name,crop,*,fixed=None,axis=0,factor=1.,tight=False,fields=False,cap=5000,repetitions=1,warmup=False,backend=None,tau=None,preflight=False):
        s=dict(self.config['settings']);s.update(return_fields=fields,wall_timeout_s=cap-30,
            n_steps_max=self.config.get('max_steps',100000),verbose=True,heartbeat=2000)
        if backend:s['backend']=backend
        if tau is not None:s['tau']=tau
        if fixed:s.update(n_steps_max=fixed,min_steps=fixed+1)
        if tight:s.update(conv_tol=s['conv_tol']/100,conv_atol=s['conv_atol']/100)
        force=[0.,0.,0.];force[axis]=self.config['force']*factor
        count=1
        for a,b in crop:count*=b-a
        return self.task(name,'solve',crop,cap*(repetitions+int(warmup))+30,settings=s,force=force,
            repetitions=repetitions,warmup=warmup,preflight=preflight,fixed_steps=bool(fixed),
            projected_disk_bytes=count*66 if fields else 20*1024**2)

    def execute(self):
        pilots=[]
        for n in self.config.get('pilot_sizes',[64,128,256]):
            crop=[[250-n//2,250+(n-n//2)]]*3
            p=self.solve(f'pilot_{n}',crop,fixed=300,fields=True,cap=600,preflight=True)
            if not p:raise RuntimeError('Staged pilot failed')
            r=json.loads((p/'result.json').read_text());pilots.append((n,r))
            if r['iterations']!=300 or r['termination_reason']!='max_steps':raise RuntimeError('Pilot unstable or incomplete')
        n,r=pilots[-1]
        bytes_per_site=max(512,r['memory']['gpu_pool_reserved_peak_sampled_bytes']/n**3*1.15)
        seconds_per_site_step=r['timing']['solve_and_diagnostics_s']/(n**3*300)
        obs=resources();candidates=[]
        for crop in self.config['candidates']:
            count=1
            for a,b in crop:count*=b-a
            estimate_memory=bytes_per_site*count
            # Budget 60k updates per required load, five loads, with 30% reserve for remaining work.
            estimate_s=seconds_per_site_step*count*60000*5 + 120
            eligible=estimate_memory<.8*obs['gpu_free_bytes'] and estimate_s<.7*self.remaining()
            candidates.append(dict(crop=crop,estimated_gpu_bytes=estimate_memory,estimated_primary_seconds=estimate_s,eligible=eligible))
        save(self.output/'selection_plan.json',dict(candidates=candidates,pilot_sizes=[p[0] for p in pilots],
             assumptions='512 bytes/site minimum or measured reservation *1.15; 60k steps/load planning, not a convergence guarantee',
             acceptance_target_relative_column=.001,acceptance_target_relative_tensor_frobenius=.001,resources=obs))
        eligible=[p for p in candidates if p['eligible']]
        if not eligible:raise RuntimeError('No candidate meets memory/runtime planning; no tolerance relaxation')
        crop=eligible[0]['crop'];self.selected=crop;self.inventory()
        p=self.solve('selected_preflight',crop,fixed=300,fields=True,cap=900,preflight=True)
        if not p:raise RuntimeError('Selected export/diagnostics preflight failed; retained, production not attempted')
        r=json.loads((p/'result.json').read_text())
        if r['iterations']!=300 or r['termination_reason']!='max_steps':raise RuntimeError('Large preflight failed')
        self.task('geometry','geometry',crop,300,projected_disk_bytes=100*1024**2)
        # Solver caps are fixed before loads; actual parent deadline always wins.
        cap=min(5000,max(600,int((self.remaining()-3600)/5)))
        baseline=[]
        for i,c in enumerate('xyz'):
            p=self.solve('baseline_'+c,crop,axis=i,fields=True,cap=cap)
            baseline.append(p)
        accepted=all(p and json.loads((p/'result.json').read_text()).get('valid_for_permeability') for p in baseline)
        if accepted:
            self.solve('half_x',crop,factor=.5,cap=cap)
            self.solve('tight_x',crop,tight=True,cap=cap)
        # Morphology profiles precede any explicit guard override at main scale.
        profiles=[]
        for n in (32,64,96):
            p=self.task(f'morphology_profile_{n}','morphology',[[250-n//2,250+n//2]]*3,300)
            if p:
                rr=json.loads((p/'result.json').read_text());profiles.append((n,rr))
        if profiles:
            n,rr=profiles[-1];count=(crop[0][1]-crop[0][0])**3
            estimate=rr['elapsed_s']*count/n**3*3 # conservative size/cache/radius allowance
            allowance=min(3600,self.remaining()-1800)
            planning=dict(profile_size=n,profile_elapsed_s=rr['elapsed_s'],profile_radii=rr['number_of_radii'],
                          estimate_s=estimate,allowance_s=allowance,explicit_size_limit=count,
                          assumption='3x linear-voxel extrapolation; parent enforces cap; radii can differ')
            save(self.output/'morphology_plan.json',planning)
            if estimate<allowance:
                self.task('morphology_main','morphology',crop,allowance,planning=planning,projected_disk_bytes=count*8+1024**2)
            else:save(self.output/'morphology_blocker.json',dict(reason='Exact full-support cost exceeds remaining allocation',**planning))
        # Performance is sequential, with no concurrent morphology/external solve.
        for n in (64,128,256,crop[0][1]-crop[0][0]):
            if self.remaining()<900:break
            bounds=[[250-n//2,250+(n-n//2)]]*3
            self.solve(f'performance_cuda_{n}',bounds,fixed=1000,cap=600,repetitions=3,warmup=True)
        for backend in ('numpy','cupy-array','cuda'):
            if self.remaining()<900:break
            self.solve('matched_'+backend,[[234,266]]*3,fixed=1000,cap=240,repetitions=3,warmup=True,backend=backend)
        if accepted and self.remaining()>2*cap+900:
            for i,c in ((1,'y'),(2,'z')):self.solve('half_'+c,crop,axis=i,factor=.5,cap=cap)
        for tau in (.8,1.,1.2):
            if self.remaining()<900:break
            self.solve(f'tau_{tau}',[[186,314]]*3,cap=600,tau=tau)

    def finalize(self):
        # This also runs after an exception, retaining incomplete/failed cases.
        self.inventory()
        crop=self.selected or self.config['candidates'][0]
        job=dict(kind='summarize',crop=crop,dataset=self.config['dataset'],dataset_shape=self.config['dataset_shape'],
                 campaign=str(self.output),output=str(self.output/'report'),fingerprint='final-report',
                 voxel_size_m=self.config['voxel_size_m'])
        save(self.output/'report_job.json',job)
        with (self.output/'report.log').open('w') as f:
            result=subprocess.run([sys.executable,str(ROOT/'benchmarks/campaign_worker.py'),str(self.output/'report_job.json')],
                                  stdout=f,stderr=subprocess.STDOUT,timeout=600)
        save(self.output/'finished.json',dict(finished=time.time(),elapsed_s=time.time()-self.started,
             report_returncode=result.returncode,source_unchanged=source_hashes()==self.sources))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--budget',type=float,default=28800);args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    campaign=Campaign(config,args.output,args.budget)
    failed=False
    try:campaign.execute()
    except Exception as error:
        failed=True
        import traceback
        save(campaign.output/'campaign_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
    finally:campaign.finalize()
    sys.exit(2 if failed else 0)
