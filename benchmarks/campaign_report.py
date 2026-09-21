"""Regenerate small evidence and figures from saved cases; never runs a solver."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time
import traceback
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from porewise.io import write_json
from porewise.tensor import assemble_tensor


def get_case(root,name):
    candidates=[]
    for path in (root/name).glob('attempt_*'):
        if (path/'result.json').exists() and (path/'complete.json').exists():candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def result(path):return json.loads((path/'result.json').read_text()) if path else None


def figure(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def tensor_report(root,out):
    tensors={};sens={};fig,ax=plt.subplots(1,2,figsize=(10,4))
    for prefix in ('baseline','half'):
        paths=[get_case(root,prefix+'_'+c) for c in 'xyz']
        loads=[result(p) for p in paths]
        if all(loads):
            t=assemble_tensor(loads,json.loads((root/'frozen.json').read_text())['config']['voxel_size_m'])
            t['public_call_wall_s_all_loads']=sum(r['public_call_wall_s'] for r in loads)
            write_json(out/(prefix+'_tensor.json'),t);tensors[prefix]=t
    base=tensors.get('baseline')
    if base and base['valid_for_permeability']:
        raw=np.array(base['K_m2']);im=ax[0].imshow(raw/1e-12,cmap='RdBu_r',vmin=-abs(raw).max()/1e-12,vmax=abs(raw).max()/1e-12)
        for i in range(3):
            for j in range(3):ax[0].text(j,i,f'{raw[i,j]/1e-12:.4g}',ha='center',va='center')
        ax[0].set(xticks=range(3),yticks=range(3),xticklabels=list('xyz'),yticklabels=list('xyz'),xlabel='Load',ylabel='Response',title='Raw permeability tensor')
        fig.colorbar(im,ax=ax[0],label='Permeability (10⁻¹² m²)')
        ax[1].bar(range(1,4),np.array(base['principal_values_m2'])/1e-12,color='teal')
        ax[1].set(xlabel='Ascending principal value',ylabel='Permeability (10⁻¹² m²)',title='Eigenvalues of symmetric part')
        figure(fig,out/'tensor')
        vec=np.array(base['principal_directions']);f=plt.figure(figsize=(5,5));a=f.add_subplot(projection='3d')
        for j in range(3):a.quiver(0,0,0,*vec[:,j],color=plt.cm.viridis(j/2),label=f'Principal {j+1}')
        a.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='x direction',ylabel='y direction',zlabel='z direction')
        a.view_init(elev=25,azim=-55);a.legend();figure(f,out/'principal_directions')
        x0=np.array(base['K_lu'])[:,0]
        for name in ('half_x','tight_x'):
            r=result(get_case(root,name))
            if r and r['valid_for_permeability']:
                column=r['nu']*np.array([r['u_'+c+'_mean_total'] for c in 'xyz'])/r['F_x']
                delta=column-x0;relative=float(np.linalg.norm(delta)/np.linalg.norm(x0))
                sens[name]=dict(relative_column_change=relative,target=.001,passed=relative<=.001,
                    absolute_component_changes_lu=delta,component_changes_over_baseline_column_norm=delta/np.linalg.norm(x0))
        half=tensors.get('half')
        if half and half['valid_for_permeability']:
            relative=float(np.linalg.norm(half['K_lu']-base['K_lu'])/np.linalg.norm(base['K_lu']))
            sens['half_tensor']=dict(relative_frobenius_change=relative,target=.001,passed=relative<=.001)
    else:plt.close(fig)
    write_json(out/'sensitivity.json',sens)
    return tensors,sens


def field_plots(root,out):
    path=get_case(root,'baseline_x');geo=get_case(root,'geometry')
    if not path or not geo or not result(path).get('valid_for_permeability'):return
    mask=np.load(geo/'blocked.npy',mmap_mode='r')
    fields={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ('ux','uy','uz')}
    meta=json.loads((path/'field_metadata.json').read_text());dx=meta['spacing_m'];origin=np.array(meta['origin_lower_corner_m'])
    speed=np.sqrt(sum(v*v for v in fields.values()));values=speed[~mask]
    counts,edges=np.histogram(values,bins=80);mass=counts/values.size
    write_json(out/'speed_distribution.json',dict(bin_edges_lu=edges,probability_mass=mass,
        weighting='equal original pore voxel volumes, including stagnant pores',units='lattice dx/dt',pore_mean_speed_lu=float(values.mean())))
    fig,ax=plt.subplots(figsize=(6,4));ax.stairs(mass,edges,color='teal')
    ax.set(xlabel='Speed (lattice units)',ylabel='Pore-volume probability per bin');figure(fig,out/'speed_distribution')
    z=mask.shape[0]//2;plane=np.ma.masked_array(speed[z],mask[z]);fig,ax=plt.subplots(figsize=(6,5))
    extent=np.array([origin[0],origin[0]+mask.shape[2]*dx,origin[1],origin[1]+mask.shape[1]*dx])*1e3
    im=ax.imshow(plane,origin='lower',extent=extent,cmap='viridis');fig.colorbar(im,ax=ax,label='Speed (lattice units)')
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'z = {(origin[2]+(z+.5)*dx)*1e3:.4g} mm; force +x')
    figure(fig,out/'speed_slice')
    from rock_streamlines import integrate
    # Streamlines are on the original full-resolution mask and field.
    seeds=[[z,y,1.5] for z in np.linspace(2,mask.shape[0]-3,10) for y in np.linspace(2,mask.shape[1]-3,10)]
    paths=[];records=[]
    for seed in seeds:
        points,why=integrate(seed,fields,mask,max_steps=3000)
        records.append(dict(seed_zyx_cell_index=seed,termination=why,points=len(points)))
        if len(points)>2:paths.append((points[:,::-1]+.5)*dx+origin)
    write_json(out/'streamlines.json',dict(seeds=records,paths_xyz_m=paths,integration='forward Euler, normalized velocity; step 0.2 voxel, segment checks every <=0.05 voxel',
        interpolation='trilinear only when all eight stencil cells are pore',max_steps=3000,
        termination='solid stencil, display boundary, speed <1e-18 LU, or maximum steps',
        interpretation='qualitative unweighted seeds, line density is not flux; no periodic continuation'))
    # VTK offscreen rendering; display-only solid subsampling does not alter path integration.
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    stride=max(1,int(np.ceil(max(mask.shape)/100)))
    display=np.array(mask[::stride,::stride,::stride],dtype='u1');display[display.shape[0]//2:]=0
    image=vtk.vtkImageData();image.SetDimensions(*display.shape[::-1]);image.SetSpacing(*([dx*stride]*3));image.SetOrigin(*(origin+.5*dx))
    image.GetPointData().SetScalars(numpy_to_vtk(display.ravel(order='C'),deep=True))
    contour=vtk.vtkFlyingEdges3D();contour.SetInputData(image);contour.SetValue(0,.5)
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(contour.GetOutputPort());mapper.ScalarVisibilityOff()
    rock=vtk.vtkActor();rock.SetMapper(mapper);rock.GetProperty().SetColor(.72,.72,.69);rock.GetProperty().SetOpacity(.45)
    renderer=vtk.vtkRenderer();renderer.SetBackground(1,1,1);renderer.AddActor(rock)
    vtkpoints=vtk.vtkPoints();lines=vtk.vtkCellArray()
    for path_points in paths:
        line=vtk.vtkPolyLine();line.GetPointIds().SetNumberOfIds(len(path_points))
        for j,p in enumerate(path_points):line.GetPointIds().SetId(j,vtkpoints.InsertNextPoint(*p))
        lines.InsertNextCell(line)
    data=vtk.vtkPolyData();data.SetPoints(vtkpoints);data.SetLines(lines)
    writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(out/'streamlines.vtp'));writer.SetInputData(data);writer.Write()
    lm=vtk.vtkPolyDataMapper();lm.SetInputData(data);actor=vtk.vtkActor();actor.SetMapper(lm)
    actor.GetProperty().SetColor(.08,.45,.55);actor.GetProperty().SetLineWidth(2);renderer.AddActor(actor)
    length=mask.shape[2]*dx
    # Cartesian axes/load arrow and a labelled physical-length scale segment.
    arrow=vtk.vtkArrowSource();am=vtk.vtkPolyDataMapper();am.SetInputConnection(arrow.GetOutputPort())
    aa=vtk.vtkActor();aa.SetMapper(am);aa.SetScale(length*.35);aa.SetPosition(*(origin+np.array([0,-.1*length,0])))
    aa.GetProperty().SetColor(.7,.1,.1);renderer.AddActor(aa)
    scale=vtk.vtkLineSource();scale.SetPoint1(*(origin+np.array([0,-.2*length,0])));scale.SetPoint2(*(origin+np.array([length/4,-.2*length,0])))
    sm=vtk.vtkPolyDataMapper();sm.SetInputConnection(scale.GetOutputPort());sa=vtk.vtkActor();sa.SetMapper(sm);sa.GetProperty().SetColor(0,0,0);sa.GetProperty().SetLineWidth(4);renderer.AddActor(sa)
    title=vtk.vtkTextActor();title.SetInput(f'Bentheimer cutaway | force +x (red arrow)\nBlack scale bar: {length/4*1e3:.3g} mm | unweighted streamlines')
    title.GetTextProperty().SetColor(0,0,0);title.GetTextProperty().SetFontSize(25);title.SetPosition(30,30);renderer.AddViewProp(title)
    camera=renderer.GetActiveCamera();centre=origin+np.array(mask.shape[::-1])*dx/2
    camera.SetPosition(*(centre+length*np.array([1.8,-2.4,1.7])));camera.SetFocalPoint(*centre);camera.SetViewUp(0,0,1)
    axes=vtk.vtkCubeAxesActor();axes.SetBounds(origin[0],origin[0]+mask.shape[2]*dx,origin[1],origin[1]+mask.shape[1]*dx,origin[2],origin[2]+mask.shape[0]*dx)
    axes.SetCamera(camera);axes.SetXTitle('x (m)');axes.SetYTitle('y (m)');axes.SetZTitle('z (m)')
    for axis in range(3):
        axes.GetTitleTextProperty(axis).SetColor(0,0,0);axes.GetLabelTextProperty(axis).SetColor(0,0,0)
    renderer.AddActor(axes)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.AddRenderer(renderer);window.SetSize(1800,1500)
    renderer.ResetCameraClippingRange();window.Render()
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
    png=vtk.vtkPNGWriter();png.SetFileName(str(out/'rock_cutaway.png'));png.SetInputConnection(capture.GetOutputPort());png.Write();window.Finalize()
    write_json(out/'render_metadata.json',dict(solid_display_stride=stride,cutaway='upper half z removed for display only',
        camera_position_xyz_m=camera.GetPosition(),focal_point_xyz_m=centre,background='white',
        streamline_colour='constant teal; no speed encoding',slice_colour='viridis linear full range; no clipping',
        original_mask_and_fields_used_for_paths=True,slice_z_index=z,scale_bar_m=length/4))


def morphology_plots(root,out):
    path=get_case(root,'morphology_main')
    if not path:return
    r=result(path);fig,ax=plt.subplots(1,2,figsize=(10,4));edges=np.array(r['bin_edges'])*1e6
    ax[0].stairs(r['probability_mass'],edges,color='teal');ax[0].set(xlabel='Local-thickness diameter (µm)',ylabel='Pore-volume probability per bin')
    ax[1].step(edges[1:],r['cumulative_undersize'],where='post',color='teal');ax[1].set(xlabel='Local-thickness diameter (µm)',ylabel='Cumulative pore-volume fraction',ylim=(0,1.01))
    figure(fig,out/'local_thickness')


def report(root):
    out=root/'report';out.mkdir(exist_ok=True);errors=[]
    tensors={};sens={}
    for label,fn in [('tensor',tensor_report),('fields',field_plots),('morphology',morphology_plots)]:
        try:
            value=fn(root,out)
            if label=='tensor':tensors,sens=value
        except Exception as e:errors.append(dict(section=label,error=repr(e),traceback=traceback.format_exc()))
    write_json(out/'plot_errors.json',errors)
    timings=[];inventory=[]
    for jobpath in sorted(root.glob('*/attempt_*/job.json')):
        p=jobpath.parent;job=json.loads(jobpath.read_text());r=result(p)
        inventory.append(dict(case=p.parent.name,attempt=p.name,completed=(p/'complete.json').exists(),
            status=r.get('termination_reason',r.get('status','ok')) if r else 'no result',
            accepted=bool(r and r.get('valid_for_permeability')),path=str(p.relative_to(root))))
        for rep in sorted(p.glob('repetition_*.json')):
            r=json.loads(rep.read_text());timings.append(dict(case=p.parent.name,attempt=p.name,repetition=rep.stem,
                backend=r['backend'],updates=r['iterations'],status=r['termination_reason'],
                public_call_wall_s=r['public_call_wall_s'],loop_s=r['timing']['solve_and_diagnostics_s'],
                setup_s=r['timing']['setup_s'],host_field_transfer_s=r['timing']['export_s'],
                total_lattice_mlups_public_call=r['total_lattice_mlups_public_call'],
                fluid_node_mlups_public_call=r['fluid_node_mlups_public_call'],
                total_lattice_mlups_iteration_loop=r['total_lattice_mlups_iteration_loop']))
    for name,rows in [('all_attempts',inventory),('timings',timings)]:
        with (out/(name+'.csv')).open('w') as f:
            if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    accepted=[r['case'] for r in inventory if r['accepted']]
    config=json.loads((root/'frozen.json').read_text())
    text='# Overnight technical handoff\n\n'
    text+=f'Generated UTC {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}. Local campaign: `{root}`.\n\n'
    text+='Source: `frozen.json` records exact package and benchmark hashes; `source/` preserves those files. Session records identify the starting Git commit and working tree. This is a new local snapshot, distinct from all historical benchmark snapshots. No archived result was relabelled.\n\n'
    text+='Accepted solve cases: '+(', '.join(accepted) or 'none')+'. See `all_attempts.csv` for failed, incomplete and fixed-step cases. Fixed-step `max_steps` states are throughput evidence, not accepted permeability.\n\n'
    text+='Raw tensors, validity, symmetric-part eigensystems and all-load wall time: `baseline_tensor.json` and, if complete, `half_tensor.json`. Sensitivity targets were predeclared at 0.1%; outcomes are in `sensitivity.json`. Failure of a target is retained without tuning it.\n\n'
    text+='Equations: nu=(tau-0.5)/3; force is force per volume with reference density one; superficial U averages the entire unchanged sample; K_ij=nu U_i/F_j and K_m2=K_lu dx². Array axes are z,y,x; tensor response rows/load columns x,y,z. Velocities are lattice units; no physical time scale has been supplied. Length 20 cells is a Reynolds diagnostic convention.\n\n'
    text+='Full accepted baseline fields are separate uncompressed float64 NPY components plus cell-centred VTI, below `baseline_*/attempt_*/`. VTI origin is the crop lower voxel-face corner; spacing is 5 µm. Large fields and morphology maps are excluded from the compact archive. Each load is independently initialized at rest; reuse requires identical source, geometry identity and settings. There are no population checkpoints; interrupted loads restart from rest.\n\n'
    text+='Morphology: `morphology_main/` is the same support if completed; otherwise `morphology_blocker.json`, failed task records and profiles explain the omission. Exact open-ball covering diameters include isolated pores and are weighted by pore voxel volume; the artificial solid exterior differs from periodic flow. Interior exclusions sample the full map, not independent ROI estimates.\n\n'
    text+='Performance: three measured repetitions after a matching warmup; `timings.csv` distinguishes synchronized public-call and iteration-loop denominators, setup and host transfer. NPY/VTI export cost is separately in field metadata. Sampled device-wide memory includes concurrent users; pool reservation is not live allocation, and RSS is process-lifetime high-water. NumPy is a reference implementation, not an optimized 96-core solver.\n\n'
    text+='Figures require no further simulation for completed inputs: raw tensor, symmetric-part principal values/directions, accepted-x speed slice and pore-volume histogram, conservative solid-safe streamlines/cutaway, and full-support PSD histogram/CDF if morphology completed. Run `.venv/bin/python benchmarks/campaign_report.py CAMPAIGN` to regenerate. `plot_errors.json` records any missing rendering. Streamline seeds are unweighted qualitative paths, not flux density.\n\n'
    text+='Limits: periodic cropped Bentheimer is an application example, not a matched portal-reference benchmark, grid-refinement study or proof of representative volume. This campaign tests force/tolerance sensitivity on this sample only. See `independent_comparison.md` for external-reference limitations. Historical analytical tests and voxel-boundary limitations remain applicable. No float32 or 500³ qualification is implied.\n'
    (out/'OVERNIGHT_HANDOFF.md').write_text(text)
    # Hash every output, retaining large-file paths separately; do not duplicate large data.
    large=[];small=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix in ('.partial',) or p.name in ('technical_evidence.tar.gz','artifact_checksums.json','large_artifacts.json'):continue
        if p.stat().st_size>16*1024**2 or p.suffix in ('.npy','.vti'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            large.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
        else:small.append(p)
    write_json(out/'large_artifacts.json',large)
    small.append(out/'large_artifacts.json')
    write_json(out/'artifact_checksums.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in small})
    small.append(out/'artifact_checksums.json')
    archive=out/'technical_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in small:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    (out/'technical_evidence.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  technical_evidence.tar.gz\n')


if __name__=='__main__':
    import sys
    report(Path(sys.argv[1]).resolve())
