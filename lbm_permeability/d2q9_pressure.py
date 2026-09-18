"""Experimental GPU-only 2D pressure solver; Zou-He planes normal to x.

Sample-face fluid-mean pressure difference divided by Nx_sample-1 is the
measured positive pressure-drop gradient. Artificial reservoirs and optional
outer wall rows are excluded from the original sample averaging volume.
"""
import time
import resource
import numpy as np
from .backends import cp, select
from . import validation as v
from .d2q9 import W, CX, CY
from .d2q9_fast import _module
from .solver import macros
from .diagnostics import Monitor


def zou_he(f, inlet_fluid, outlet_fluid, rho_in, rho_out, xp):
    """Reconstruct incoming populations; prescribed density and zero tangential velocity."""
    # inlet x=0: unknown incoming q1,q5,q8 (cx=+1); rho fixed, u_y=0
    c = f[:, :, 0]
    C = c[0] + c[3] + c[4]
    B = c[2] + c[6] + c[7]
    ux = 1.0 - (C + 2.0 * B) / rho_in
    f1 = c[2] + (2.0 / 3.0) * rho_in * ux
    half = 0.5 * (c[3] - c[4])
    f5 = c[7] - half + (1.0 / 6.0) * rho_in * ux
    f8 = c[6] + half + (1.0 / 6.0) * rho_in * ux
    f[1, :, 0] = xp.where(inlet_fluid, f1, c[1])
    f[5, :, 0] = xp.where(inlet_fluid, f5, c[5])
    f[8, :, 0] = xp.where(inlet_fluid, f8, c[8])
    # outlet x=Nx-1: unknown incoming q2,q6,q7 (cx=-1); rho fixed, u_y=0
    d = f[:, :, -1]
    C = d[0] + d[3] + d[4]
    A = d[1] + d[5] + d[8]
    ux = -1.0 + (C + 2.0 * A) / rho_out
    f2 = d[1] - (2.0 / 3.0) * rho_out * ux
    half = 0.5 * (d[3] - d[4])
    f6 = d[8] - half - (1.0 / 6.0) * rho_out * ux
    f7 = d[5] + half - (1.0 / 6.0) * rho_out * ux
    f[2, :, -1] = xp.where(outlet_fluid, f2, d[2])
    f[6, :, -1] = xp.where(outlet_fluid, f6, d[6])
    f[7, :, -1] = xp.where(outlet_fluid, f7, d[7])

def lbm_stokes_2d_pressure(blocked, deltaP=1e-4, tau=1., pad=4, walls_y=False,
                           n_steps_max=400000, energy_eps=1e-4, energy_window=1000,
                           sample_every=50, precision='float64', verbose=True,
                           heartbeat=5000, return_fields=True, *, wall_timeout_s=None,
                           conv_atol=1e-12, consecutive=3, stability_every=50,
                           flux_tol=1e-4, max_mach=.05, characteristic_length=1.):
    started=time.perf_counter()
    blocked=v.mask(blocked,2)
    v.parameters(tau,(deltaP,),n_steps_max,energy_eps,sample_every,precision,wall_timeout_s,
                 heartbeat,energy_window=energy_window,stability_every=stability_every,consecutive=consecutive)
    v.integer('pad',pad,0)
    for name,val in dict(conv_atol=conv_atol,flux_tol=flux_tol,max_mach=max_mach,characteristic_length=characteristic_length).items(): v.positive(name,val)
    if deltaP==0 or 1.-3.*deltaP<=0:
        raise ValueError('deltaP must be nonzero and give positive outlet density (1-3*deltaP)')
    Ny0,Nx0=blocked.shape
    if Nx0<2 or blocked[:,0].all() or blocked[:,-1].all():
        raise ValueError('sample needs two distinct x faces each containing fluid')
    if not blocked.any() and not walls_y:
        raise ValueError('fully fluid pressure channel needs confining walls to define resistance')
    select('cuda')
    blk=np.pad(blocked,((0,0),(pad,pad)),constant_values=False)
    if walls_y: blk=np.pad(blk,((1,1),(0,0)),constant_values=True)
    y_off=int(walls_y)
    Ny,Nx=blk.shape
    bd=cp.asarray(blk);fluid=~bd;solid=bd.astype(cp.uint8)
    fa=cp.empty((9,Ny,Nx),dtype=precision);fb=cp.empty_like(fa)
    for q in range(9): fa[q]=W[q]
    mod=_module(precision);collide=mod.get_function('collide');stream=mod.get_function('stream')
    blocks=((blk.size+255)//256,)
    ac=(solid,np.int64(blk.size),np.float64(0),np.float64(0),np.float64(tau),np.float64(0))
    ast=(solid,np.int32(Nx),np.int32(Ny))
    nu=(tau-.5)/3;rho_in=1.;rho_out=1.-3.*deltaP
    inlet,outlet=fluid[:,0],fluid[:,-1]
    c_in,c_out=pad,pad+Nx0-1
    roi=(slice(y_off,y_off+Ny0),slice(c_in,c_out+1))
    mon=Monitor(cp,energy_eps,conv_atol,consecutive,flux_tol,periodic=False)
    previous_gradient=None;previous_flux=None
    def inspect(step):
        nonlocal previous_gradient,previous_flux
        fields,rho=macros(fa,bd,(0.,0.),(CX,CY),cp)
        ux,uy=fields
        p_in=float(rho[:,c_in][fluid[:,c_in]].mean())/3
        p_out=float(rho[:,c_out][fluid[:,c_out]].mean())/3
        gradient=(p_in-p_out)/(Nx0-1)
        flux=cp.sum(rho*ux,axis=0,dtype=cp.float64)
        q0,q1=float(flux[0]),float(flux[-1])
        scale=max(abs(q0),abs(q1),conv_atol*Ny)
        imbalance=abs(q0-q1)/scale
        span=float(flux.max()-flux.min())/scale
        stable=False
        if previous_gradient is not None:
            stable=abs(gradient-previous_gradient)<=conv_atol+energy_eps*abs(gradient)
            stable=stable and np.max(abs(np.array([q0,q1])-previous_flux))<=conv_atol*Ny+energy_eps*scale
        previous_gradient=gradient;previous_flux=np.array([q0,q1])
        extra=dict(k_lu_diagnostic_estimate=nu*float(ux[roi].mean())/gradient if gradient!=0 else None, pressure_drop_gradient=gradient,flux_in=q0,flux_out=q1,flux_imbalance=imbalance,
                   flux_span_relative=span,pressure_checks_passed=stable and gradient*deltaP>0 and span<=flux_tol)
        status,diag=mon.check(step,fields,rho,fa,fluid,nu,characteristic_length,extra)
        return fields,rho,gradient,flux,status,diag
    cp.cuda.get_current_stream().synchronize()
    setup_s=time.perf_counter()-started;solve_start=time.perf_counter()
    reason='max_steps';iterations=0;peak_pool=0
    for step in range(1,n_steps_max+1):
        if wall_timeout_s is not None and time.perf_counter()-started>=wall_timeout_s:
            reason='timeout';break
        collide(blocks,(256,),(fa,fb)+ac);stream(blocks,(256,),(fb,fa)+ast)
        zou_he(fa,inlet,outlet,rho_in,rho_out,cp)
        iterations=step
        if step%stability_every==0:
            rho=fa.sum(axis=0,dtype=cp.float64)
            if not bool(cp.isfinite(fa).all()): reason='nonfinite';break
            if bool((rho[fluid]<=0).any()): reason='invalid_density';break
        if step%sample_every==0:
            fields,rho,gradient,flux,status,diag=inspect(step)
            peak_pool=max(peak_pool,cp.get_default_memory_pool().total_bytes())
            if status and (status!='converged' or step>=energy_window): reason=status;break
        if verbose and step%heartbeat==0: print(f'  pressure: {step} updates',flush=True)
    cp.cuda.get_current_stream().synchronize();solve_s=time.perf_counter()-solve_start
    final_diagnostics_start=time.perf_counter()
    if reason in ('nonfinite','invalid_density') or not mon.history or mon.history[-1]['iterations']!=iterations:
        fields,rho,gradient,flux,status,diag=inspect(iterations)
        if status in ('nonfinite','invalid_density'): reason=status
    if not bool(cp.isfinite(fa).all()) or not all(bool(cp.isfinite(u).all()) for u in fields): reason='nonfinite'
    elif bool((rho[fluid]<=0).any()): reason='invalid_density'
    if reason=='converged' and diag.get('mach_max',np.inf)>max_mach: reason='quality_failed'
    accepted=reason=='converged'
    means=[float(u[roi].mean()) for u in fields]
    estimate=nu*means[0]/gradient if gradient!=0 and np.isfinite(gradient) else None
    out=dict(k_lu=estimate if accepted else None,k_lu_diagnostic_estimate=estimate,
             nu=nu,deltaP=deltaP,u_x_mean_total=means[0],u_y_mean_total=means[1],
             converged=accepted,valid_for_permeability=accepted,termination_reason=reason,
             iterations=iterations,iterations_completed=iterations,step_converged=iterations if accepted else None,
             convergence_history=mon.history,diagnostics=diag,backend='cuda',experimental=True,
             storage_dtype=precision,compute_dtype='float64',diagnostic_dtype='float64',precision=precision,
             shape=list(blocked.shape),array_axes=['y','x'],components=['x','y'],
             Nx_padded=Nx,pad=pad,walls_y=walls_y,porosity=float((~blocked).mean()),
             boundary_conditions='Zou-He density x planes; periodic y with optional outer solid rows',
             force_convention='positive signed pressure-drop gradient; mu_ref=nu, rho_ref=1',
             settings=dict(tau=tau,deltaP=deltaP,pad=pad,walls_y=walls_y,n_steps_max=n_steps_max,
                           energy_eps=energy_eps,energy_window=energy_window,sample_every=sample_every,
                           consecutive=consecutive,conv_atol=conv_atol,flux_tol=flux_tol,
                           wall_timeout_s=wall_timeout_s,stability_every=stability_every,max_mach=max_mach,
                           characteristic_length=characteristic_length),
             pressure_profile=cp.asnumpy(cp.sum(rho*fluid,axis=0)/cp.maximum(fluid.sum(axis=0),1)/3),
             pressure_profile_fluid_counts=cp.asnumpy(fluid.sum(axis=0)),
             mass_flux_profile=cp.asnumpy(flux),compiler_options=[])
    # Empty internal sections have no defined fluid pressure; retain explicit NaN
    # in arrays (JSON exporter maps it to null), never use these for face gradient.
    out['pressure_profile'][out['pressure_profile_fluid_counts']==0]=np.nan
    out['k_history']=np.array([(d['iterations'],d['k_lu_diagnostic_estimate']) for d in mon.history])
    out['k_history_quantity']='diagnostic permeability in lattice units; acceptance requires valid_for_permeability'
    final_diagnostics_s=time.perf_counter()-final_diagnostics_start
    export_start=time.perf_counter()
    if return_fields:
        out.update(ux=cp.asnumpy(fields[0][roi]),uy=cp.asnumpy(fields[1][roi]),rho=cp.asnumpy(rho[roi]))
    out.update(elapsed_s=time.perf_counter()-started,
               timing=dict(setup_s=setup_s,final_diagnostics_s=final_diagnostics_s,solve_and_diagnostics_s=solve_s,export_s=time.perf_counter()-export_start),
               memory=dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                           gpu_pool_reserved_peak_sampled_bytes=max(peak_pool,cp.get_default_memory_pool().total_bytes())))
    return out
