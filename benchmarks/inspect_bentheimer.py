"""Validate selected DRP-29 bytes/labels and create predetermined crops and previews."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.io import write_json,save_result
from porewise.morphology import local_thickness_distribution

root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot/dataset')
p=root/'Seg_Oxyz_0001_0001_0001.raw'
assert p.stat().st_size==500**3
s=np.memmap(p,dtype='u1',shape=(500,500,500),mode='r')
values,counts=np.unique(s,return_counts=True)
assert np.array_equal(values,[0,255]),(values,counts)
report=dict(segmentation_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),
            encoding=dict(pore=0,solid=255),labels=dict(zip(map(str,values),map(int,counts))),
            full_porosity=float(np.count_nonzero(s==0)/s.size),
            array_axes=['z','y','x'],voxel_size_m=5e-6,crops={},
            orientation_note='Raw width/height/stack mapping; no claim about physical handedness')
for n in (128,256):
    m=s[:n,:n,:n]==255
    np.save(root/f'blocked_{n}.npy',m)
    report['crops'][str(n)]=dict(bounds_zero_based_half_open=[[0,n]]*3,porosity=float((~m).mean()),
                               mask_sha256=hashlib.sha256(m.tobytes()).hexdigest())
ct=root/'Cropped_Oxyz_001_001_001_Nxyz_500_500_500.raw'
if ct.exists():
    assert ct.stat().st_size==2*500**3
    image=np.memmap(ct,dtype='<u2',shape=s.shape,mode='r')
    # Deterministic sparse sample limits temporary allocation.
    sampled=image[::5,::5,::5];lab=s[::5,::5,::5]
    report['ct_alignment']=dict(mean_intensity_pore=float(sampled[lab==0].mean()),mean_intensity_solid=float(sampled[lab==255].mean()),
                                note='Grayscale correlation supports segmentation alignment; it does not prove physical axis orientation')
for name in ('Xvelocity','Yvelocity','Zvelocity','Pressure'):
    f=root/(name+'_Oxyz_0001_0001_0001.raw')
    if not f.exists():continue
    assert f.stat().st_size==4*500**3
    a=np.memmap(f,dtype='<f4',shape=s.shape,mode='r')
    b=a[::5,::5,::5];solid=s[::5,::5,::5]==255
    report[name]=dict(finite=bool(np.isfinite(b).all()),sample_min=float(b.min()),sample_max=float(b.max()),
                      solid_nonzero_fraction=float(np.count_nonzero(b[solid])/solid.sum()),
                      pore_nonzero_fraction=float(np.count_nonzero(b[~solid])/(~solid).sum()),
                      units='not recovered; not used for permeability comparison')
write_json(root/'inspection.json',report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(12,4))
for axis,ax in enumerate(axs):
    ax.imshow(np.take(s,64,axis=axis),cmap='gray',vmin=0,vmax=255)
    ax.set_title(f'array {"zyx"[axis]}=64; black=pore')
fig.tight_layout();fig.savefig(root/'segmentation_slices.png',dpi=150);plt.close(fig)
roi=s[:64,:64,:64]==255
out=root/'morphology_64'
if not (out/'result.json').exists():
    result=local_thickness_distribution(roi,voxel_size=5e-6,return_map=True)
    save_result(out,result,dict(bounds_zero_based_half_open=[[0,64]]*3,parent_segmentation_sha256=report['segmentation_sha256']))
print(json.dumps(report,indent=2))
