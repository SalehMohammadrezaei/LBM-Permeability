"""Shared periodic solver lifecycle; original BGK/Guo and solid-node reflection."""
import time
import numpy as np
from . import validation as v
from .backends import select, cp
from .diagnostics import Monitor
from .memory import process_memory_report


def lattice(ndim):
    if ndim == 2:
        from .d2q9 import CX, CY, W, OPP
        return (CX,CY), W, [(q,int(OPP[q])) for q in range(1,9) if q < OPP[q]]
    from .d3q19 import CX,CY,CZ,W,OPP_PAIRS
    return (CX,CY,CZ),W,OPP_PAIRS


def macros(f, blocked, force, velocities, xp):
    # Accumulate diagnostics in double even with float32 population storage.
    rho = f.sum(axis=0, dtype=xp.float64)
    fields=[]
    for c,F in zip(velocities,force):
        mom=xp.full(blocked.shape,.5*F,dtype=xp.float64)
        for q in range(len(c)):
            if c[q]:
                mom += int(c[q])*f[q].astype(xp.float64,copy=False)
        # Do not replace invalid fluid density: checks reject it explicitly.
        fields.append(xp.where(blocked, 0., mom/rho))
    return tuple(fields),rho


def periodic(blocked, force, *, tau=1., n_steps_max=50000, conv_tol=1e-5,
             conv_window=200, backend='auto', precision='float64', return_fields=True,
             verbose=True, heartbeat=1000, wall_timeout_s=None, stability_every=50,
             conv_atol=1e-12, consecutive=3, mass_tol=None, max_mach=.05,
             characteristic_length=1., min_steps=None, mempool_flush=2000,
             collision='bgk', magic=None):
    started=time.perf_counter()
    blocked=v.mask(blocked,len(force))
    v.parameters(tau,force,n_steps_max,conv_tol,conv_window,precision,wall_timeout_s,
                 heartbeat,mempool_flush,stability_every=stability_every,consecutive=consecutive)
    v.positive('conv_atol',conv_atol); v.positive('max_mach',max_mach)
    v.positive('characteristic_length',characteristic_length)
    min_steps=conv_window*consecutive if min_steps is None else v.integer('min_steps',min_steps)
    mass_tol=(1e-8 if precision=='float64' else 1e-4) if mass_tol is None else v.positive('mass_tol',mass_tol)
    if collision not in ('bgk','trt'):
        raise ValueError("collision must be 'bgk' or 'trt'")
    if collision=='bgk':
        if magic is not None: raise ValueError('magic applies to the trt collision only')
    else:
        magic=v.positive('magic',3/16 if magic is None else magic)
    # TRT: the symmetric rate sets the viscosity, the antisymmetric rate follows
    # from magic=(1/w_plus-1/2)(1/w_minus-1/2); 3/16 fixes the bounce-back wall half-way.
    om_p=1/tau
    om_m=1/(.5+magic/(tau-.5)) if collision=='trt' else om_p
    backend,xp=select(backend)
    sparse=backend=='cuda-sparse'
    if sparse and blocked.ndim!=3:
        raise ValueError('cuda-sparse stores D3Q19 pore nodes; use cuda for 2D masks')
    if backend not in ('cuda','cuda-sparse') and precision != 'float64':
        raise ValueError('array reference backends support float64 only; use cuda for float32 storage')
    nu=(tau-.5)/3
    ndim=blocked.ndim
    base=dict(nu=nu,collision=collision,magic=magic,omega_plus=om_p,omega_minus=om_m,backend=backend,precision=precision,storage_dtype=precision,compute_dtype='float64',
              diagnostic_dtype='float64',boundary_conditions='periodic in every array axis',
              array_axes=list('yx' if ndim==2 else 'zyx'),components=list('xyz'[:ndim]),
              shape=list(blocked.shape),porosity=float((~blocked).mean()),
              force_convention='force density; rho_ref=1; mu_ref=nu',rho_ref=1.,
              settings=dict(tau=tau,force=list(force),n_steps_max=n_steps_max,conv_tol=conv_tol,
                            conv_atol=conv_atol,conv_window=conv_window,consecutive=consecutive,
                            stability_every=stability_every,wall_timeout_s=wall_timeout_s,
                            min_steps=min_steps,mass_tol=mass_tol,max_mach=max_mach,
                            characteristic_length=characteristic_length,
                            collision=collision,magic=magic),
              compiler_options=[] if backend in ('cuda','cuda-sparse') else None)
    base.update({f'F_{c}':float(F) for c,F in zip('xyz',force)})
    special='all_solid' if blocked.all() else ('zero_forcing' if not np.any(force) else ('fully_fluid_periodic' if not blocked.any() else None))
    if special:
        valid=special=='all_solid' and bool(np.any(force))
        base.update(converged=valid,valid_for_permeability=valid,termination_reason=special,
                    iterations=0,iterations_completed=0,step_converged=0 if valid else None,
                    convergence_history=[],diagnostics={},elapsed_s=time.perf_counter()-started,
                    k_lu=0. if valid else None)
        for c in 'xyz'[:ndim]:
            base[f'u_{c}_mean_total']=0.
            if return_fields: base['u'+c]=np.zeros(blocked.shape)
        return base
    force_scale=float(np.max(np.abs(force)))
    base['force_to_storage_epsilon']=force_scale/np.finfo(precision).eps
    if force_scale < np.finfo(precision).eps:
        raise ValueError('force is below population storage roundoff; increase it or use float64 (above this guard accuracy still needs verification)')
    c,w,pairs=lattice(ndim)
    opp=list(range(len(w)))
    for a,b in pairs: opp[a],opp[b]=b,a
    peak_pool=0;share=None
    if sparse:
        from .d3q19_sparse import SparseD3Q19
        state=SparseD3Q19(blocked,force,om_p,om_m,precision)
        f=state.f;fluid=slice(None);share=base['porosity']
        look=lambda:state.macros()
    else:
        bd=xp.asarray(blocked); fluid=~bd
        f=xp.empty((len(w),)+blocked.shape,dtype=precision)
        for q in range(len(w)): f[q]=w[q]
        look=lambda:macros(f,bd,force,c,xp)
    if backend=='cuda':
        if ndim==2:
            from .d2q9_fast import _module
        else:
            from .d3q19_fast import _module
        mod=_module(precision)
        collide,stream=mod.get_function('collide_trt' if collision=='trt' else 'collide'),mod.get_function('stream')
        fb=xp.empty_like(f); solid=xp.asarray(blocked,dtype=xp.uint8)
        blocks=((blocked.size+255)//256,)
        args_c=(solid,np.int64(blocked.size),*(np.float64(a) for a in force),np.float64(tau),np.float64(1-.5/tau))
        if collision=='trt': args_c += (np.float64(om_m),np.float64(1-.5*om_m))
        args_s=(solid,*(np.int32(n) for n in blocked.shape[::-1]))
    if xp is not np: xp.cuda.get_current_stream().synchronize()
    setup_s=time.perf_counter()-started
    mon=Monitor(xp,conv_tol,conv_atol,consecutive,mass_tol,pore_fraction=share)
    mon.initial_mass=float(state.count if sparse else blocked.size)
    fields,rho=look()
    mon.check(0,fields,rho,f,fluid,nu,characteristic_length)
    reason='max_steps'; iterations=0; diagnostics={}
    initial_diagnostics_s=time.perf_counter()-started-setup_s
    solve_start=time.perf_counter()
    for step in range(1,n_steps_max+1):
        if wall_timeout_s is not None and time.perf_counter()-started >= wall_timeout_s:
            reason='timeout'; break
        if sparse:
            state.step();f=state.f
        elif backend=='cuda':
            collide(blocks,(256,),(f,fb)+args_c)
            stream(blocks,(256,),(fb,f)+args_s)
        else:
            fields,rho=macros(f,bd,force,c,xp)
            u2=sum(u*u for u in fields)
            uf=sum(u*F for u,F in zip(fields,force))
            for q in range(len(w)):
                cu=sum(int(cc[q])*u for cc,u in zip(c,fields))
                cf=sum(int(cc[q])*F for cc,F in zip(c,force))
                eq=w[q]*rho*(1+3*cu+4.5*cu*cu-1.5*u2)
                source=w[q]*(1-.5/tau)*(3*cf+9*cu*cf-3*uf)
                if collision=='bgk':
                    f[q] += (-(f[q]-eq)/tau+source)*fluid
                    continue
                if q>opp[q]: continue
                # even/odd parts of the pair (q, opposite q); q==0 is its own opposite
                a,b=f[q],f[opp[q]]
                even=om_p*(.5*(a+b)-w[q]*rho*(1+4.5*cu*cu-1.5*u2))-(1-.5*om_p)*w[q]*(9*cu*cf-3*uf)
                odd=om_m*(.5*(a-b)-w[q]*rho*3*cu)-(1-.5*om_m)*w[q]*3*cf
                if q==opp[q]:
                    f[q] -= even*fluid
                else:
                    f[q],f[opp[q]]=a-(even+odd)*fluid,b-(even-odd)*fluid
            for q in range(len(w)):
                f[q]=xp.roll(f[q],tuple(int(cc[q]) for cc in c[::-1]),axis=tuple(range(ndim)))
            for a,b in pairs:
                temp=xp.where(bd,f[b],f[a])
                f[b]=xp.where(bd,f[a],f[b]); f[a]=temp
        iterations=step
        if step%stability_every==0 or step%conv_window==0 or step==n_steps_max:
            if step%conv_window==0:
                # The full monitor performs population/density validity checks.
                # Do not repeat them or construct unused velocities at the
                # intermediate stability-only checks.
                fields,rho=look()
                status,diagnostics=mon.check(step,fields,rho,f,fluid,nu,characteristic_length)
                if status and (status!='converged' or step>=min_steps):
                    reason=status; break
            else:
                rho=f.sum(axis=0,dtype=xp.float64)
                if not bool(xp.isfinite(f).all()) or not bool(xp.isfinite(rho).all()):
                    reason='nonfinite'; break
                if bool((rho[fluid]<=0).any()):
                    reason='invalid_density'; break
            if xp is not np: peak_pool=max(peak_pool,xp.get_default_memory_pool().total_bytes())
        if verbose and step%heartbeat==0:
            print(f'  {backend}: {step}/{n_steps_max} updates, {time.perf_counter()-started:.2f}s',flush=True)
    if xp is not np: xp.cuda.get_current_stream().synchronize()
    solve_s=time.perf_counter()-solve_start
    final_diagnostics_start=time.perf_counter()
    fields,rho=look()
    if reason in ('nonfinite','invalid_density') or not mon.history or mon.history[-1]['iterations']!=iterations:
        status,diagnostics=mon.check(iterations,fields,rho,f,fluid,nu,characteristic_length)
        if status in ('nonfinite','invalid_density'): reason=status
    elif mon.history:
        diagnostics=mon.history[-1]
    # Final mandatory validation even after a reported steady check.
    if not bool(xp.isfinite(f).all()) or not all(bool(xp.isfinite(u).all()) for u in fields): reason='nonfinite'
    elif bool((rho[fluid]<=0).any()): reason='invalid_density'
    if reason=='converged' and diagnostics.get('mach_max',np.inf)>max_mach: reason='quality_failed'
    accepted=reason=='converged'
    base.update(converged=accepted,valid_for_permeability=accepted,termination_reason=reason,
                iterations=iterations,iterations_completed=iterations,step_converged=iterations if accepted else None,
                convergence_history=mon.history,diagnostics=diagnostics)
    for component,u in zip('xyz',fields):
        base[f'u_{component}_mean_total']=float(u.mean(dtype=xp.float64))*(1. if share is None else share)
    loads=np.flatnonzero(force)
    estimate=(base[f'u_{"xyz"[loads[0]]}_mean_total']*nu/force[loads[0]]) if len(loads)==1 else None
    base['k_lu']=estimate if accepted else None
    base['k_lu_diagnostic_estimate']=estimate
    final_diagnostics_s=time.perf_counter()-final_diagnostics_start
    export_start=time.perf_counter()
    if return_fields:
        if sparse:
            for component,u in zip('xyz',fields): base['u'+component]=state.dense(u)
            base['rho']=np.where(blocked,1.,state.dense(rho))
        else:
            for component,u in zip('xyz',fields): base['u'+component]=np.asarray(u) if xp is np else cp.asnumpy(u)
            base['rho']=np.asarray(rho) if xp is np else cp.asnumpy(rho)
    if xp is not np:
        peak_pool=max(peak_pool,xp.get_default_memory_pool().total_bytes())
    base.update(elapsed_s=time.perf_counter()-started,
                timing=dict(setup_s=setup_s,initial_diagnostics_s=initial_diagnostics_s,final_diagnostics_s=final_diagnostics_s,solve_and_diagnostics_s=solve_s,loop_note='iteration loop includes scheduled checks; initial/final checks reported separately',export_s=time.perf_counter()-export_start),
                memory=dict(**process_memory_report(),
                            gpu_pool_reserved_peak_sampled_bytes=peak_pool,
                            **(dict(sparse_bytes=state.bytes()) if sparse else {}),
                            note='RSS is process lifetime high-water; GPU is sampled allocator reservation including cache'))
    return base
