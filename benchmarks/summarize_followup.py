"""Machine-readable follow-up summary from executed results only."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot')
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else None
out=dict(scope='technical evidence; periodic cropped-rock application, no matched external reference',
 authorization='owner approved longer than 60 minutes if needed after convergence/easy optimization review')
out['tolerance_checks']={str(n):read(f'convergence_review/tolerance_{n}.json') for n in (128,256)}
out['force_sensitivity']=read('long_rock/force_sensitivity.json')
out['tensors']={}
for i in (0,1):
 r=read(f'long_rock/tensor_force_{i}/result.json')
 if r:
  out['tensors'][str(i)]={k:r.get(k) for k in ('valid_for_permeability','K_lu','K_m2','reciprocity_error','principal_values_m2','principal_directions','elapsed_s')}
  out['tensors'][str(i)]['loads']=[{k:x.get(k) for k in ('iterations','termination_reason','elapsed_s','diagnostics','memory','timing')} for x in r['loads']]
out['end_to_end']=read('end_to_end/summary.json')
if len(out['tensors'])==2 and out['force_sensitivity'] and out['force_sensitivity']['valid_pair']:
 k0=np.array(out['tensors']['0']['K_lu']);k1=np.array(out['tensors']['1']['K_lu'])
 out['force_sensitivity']['absolute_entry_changes_lu']=abs(k1-k0).tolist()
 out['force_sensitivity']['relative_column_changes']=(np.linalg.norm(k1-k0,axis=0)/np.linalg.norm(k0,axis=0)).tolist()
out['whole_process_cli']=read('cli_timing/summary.json')
out['accepted_velocity_export']=read('accepted_128_fields/export_evidence.json')
tests={}
for name in ('review_tests','installed_cpu_review_tests','release_tests',
             'installed_cpu_release_tests','installed_gpu_release_tests'):
 p=root/(name+'.xml')
 if p.exists():
  e=ET.parse(p).getroot().find('testsuite');tests[name]=e.attrib
out['tests']=tests
entries=[]
for name,key in [('convergence_review/overhead_comparison.json','charged_seconds'),
 ('convergence_review/tolerance_128.json','charged_seconds'),('convergence_review/tolerance_256.json','charged_seconds'),
 ('convergence_review/solid_shortcut_probe.json','charged_seconds'),('end_to_end/summary.json','charged_seconds'),
 ('long_rock/budget.json','charged_seconds'),('convergence_review/stream_wrap_probe.json','charged_seconds'),
 ('cli_timing/summary.json','charged_seconds'),('accepted_128_fields/export_evidence.json','charged_seconds')]:
 r=read(name)
 if r:entries.append(dict(file=name,seconds=r[key]))
for name,r in tests.items():entries.append(dict(file=name+'.xml',seconds=float(r['time'])))
out['recorded_followup_cost']=dict(entries=entries,sum_seconds=sum(x['seconds'] for x in entries),
 note='sum of recorded suite and simulation-call wall times, including warmups; some CPU tests overlap GPU work; re-used x load not charged twice; preprocessing/install/export excluded')
probe=read('convergence_review/solid_shortcut_probe.json')
if probe:
 out['experimental_solid_shortcut']=[]
 for r in probe['cases']:
  times={label:[x['wall_s'] for x in r['repetitions'] if x['variant']==label and not x['warmup']] for label in ('before','shortcut')}
  med={k:float(np.median(v)) for k,v in times.items()}
  out['experimental_solid_shortcut'].append(dict(shape=r['shape'],storage_dtype=r['precision'],
   median_loop_wall_s=med,fractional_loop_time_reduction=1-med['shortcut']/med['before'],max_population_difference=r['max_population_difference']))
path=root/'followup_summary.json';path.write_text(json.dumps(out,indent=2)+'\n');print(path)
