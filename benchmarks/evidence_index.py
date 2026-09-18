"""Index saved evidence without running simulations; retain invalid cases."""
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot')
rows=[]
for path in sorted(root.rglob('result.json')):
 r=json.loads(path.read_text())
 # Tensor exports repeat their loads; retain paths but label duplicate entries.
 duplicate=path.parent.name.startswith('load_') and (path.parent.parent/'result.json').exists()
 rows.append(dict(path=str(path),duplicate_tensor_load=duplicate,
  valid=r.get('valid_for_permeability'),status=r.get('termination_reason','tensor' if 'K_lu' in r else 'morphology'),
  iterations=r.get('iterations'),elapsed_s=r.get('elapsed_s'),backend=r.get('backend'),
  storage_dtype=r.get('storage_dtype'),compute_dtype=r.get('compute_dtype'),
  k_lu=r.get('k_lu'),mach_max=r.get('diagnostics',{}).get('mach_max'),
  mass_drift=r.get('diagnostics',{}).get('mass_drift')))
with (root/'case_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
failed=[r for r in rows if r['valid'] is False and not r['duplicate_tensor_load']]
(root/'excluded_results.json').write_text(json.dumps(dict(
 note='Invalid for permeability, not necessarily a failed test: fixed-step probes deliberately stop at max_steps. See individual case configuration. No invalid permeability is promoted to an accepted result.',
 cases=failed),indent=2)+'\n')
print(f'{len(rows)} indexed files; {len(failed)} invalid non-duplicate result entries')
