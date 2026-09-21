"""Preserve discrepancies to legacy series without certifying reference conventions."""
import sys,json,csv,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.cylinder_array import sangani_acrivos_square
from validation.sphere_array import sangani_acrivos_sc
from benchmarks.run_cases import generate
from porewise.io import write_json
import numpy as np
root=Path('results/2026-09-18-pilot/verification');rows=[]
for kind in ('sphere','cylinder'):
 for n in (16,24,32):
  r=json.loads((root/f'{kind}_{n}/result.json').read_text());m=generate(r['metadata']['case']['geometry']);c=float(m.mean())
  radius=n*(c*3/(4*np.pi))**(1/3) if kind=='sphere' else n*np.sqrt(c/np.pi)
  reference=sangani_acrivos_sc(c)[0] if kind=='sphere' else sangani_acrivos_square(c)
  k=float(np.mean(np.diag(r['K_lu']))) if kind=='sphere' else r['k_lu']
  measured=k/radius**2
  rows.append(dict(case=f'{kind}_{n}',actual_solid_fraction=c,volume_equivalent_radius=radius,
                   measured_k_over_a2=measured,legacy_formula_k_over_a2=reference,
                   signed_relative_discrepancy=measured/reference-1,
                   accepted_independent_reference=False,reason='primary coefficient/velocity normalization not recovered; legacy truncated correlation only'))
write_json(root/'legacy_correlation_comparisons.json',rows)
print(json.dumps(rows,indent=2))
