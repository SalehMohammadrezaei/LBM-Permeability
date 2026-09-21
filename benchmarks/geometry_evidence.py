"""Small geometric degeneracies: preserve masks and explicit flow/link behavior."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.solver import periodic
from porewise.topology import connectivity_report
from porewise.io import save_result,write_json
root=Path('results/2026-09-18-pilot/geometry');root.mkdir(exist_ok=True);cost=0.
closed=np.ones((12,16),bool);closed[3:9,4:12]=False
slab=np.zeros((12,16),bool);slab[:,7:9]=True
channel=np.ones((12,16),bool);channel[3:9,:]=False
diagonal=~np.eye(12,dtype=bool)
face_pair=np.ones((8,8),bool);face_pair[3,0]=face_pair[3,-1]=False
for name,m in [('closed_cavity',closed),('blocked_x_slab',slab),('resolved_x_channel',channel),('diagonal_only_loop',diagonal),('touching_faces_without_winding',face_pair)]:
 t=time.perf_counter();r=periodic(m,(1e-6,0.),backend='numpy',verbose=False,n_steps_max=15000,conv_window=50,conv_tol=1e-6,wall_timeout_s=15,return_fields=True);cost+=time.perf_counter()-t
 info={c:connectivity_report(m,connectivity=c,boundary='periodic') for c in ('face','lattice')}
 save_result(root/name,r,dict(topology=info,interpretation='D2Q9 diagonal transport is allowed by its links; face disconnection alone does not certify zero discrete LBM flow'))
 np.save(root/name/'mask.npy',m)
 print(name,r['termination_reason'],r['k_lu'],flush=True)
write_json(root/'budget.json',dict(charged_seconds=cost))
