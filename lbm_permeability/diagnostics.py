"""Common full-vector/field convergence and conservation monitoring."""
import numpy as np


class Monitor:
    def __init__(self, xp, tol, atol, consecutive, mass_tol, periodic=True, pore_fraction=None):
        # pore_fraction: fields hold fluid nodes only; rescale their means to the total volume
        self.pore_fraction = pore_fraction
        self.xp, self.tol, self.atol = xp, tol, atol
        self.required, self.mass_tol, self.periodic = consecutive, mass_tol, periodic
        self.previous = None
        self.previous_means = None
        self.initial_mass = None
        self.streak = 0
        self.history = []

    def check(self, step, fields, rho, f, fluid, nu, length, extra=None):
        xp = self.xp
        if not bool(xp.isfinite(f).all()) or not bool(xp.isfinite(rho).all()):
            return 'nonfinite', {'iterations': step, 'finite_state': False, 'reason': 'nonfinite populations or macroscopic fields'}
        if bool((rho[fluid] <= 0).any()):
            return 'invalid_density', {'iterations': step, 'finite_state': True, 'rho_min': float(rho[fluid].min()), 'reason': 'nonpositive fluid density'}
        if not all(bool(xp.isfinite(u).all()) for u in fields):
            return 'nonfinite', {'iterations': step, 'finite_state': False, 'reason': 'nonfinite populations or macroscopic fields'}
        mass = float(f.sum(dtype=xp.float64))
        if self.initial_mass is None:
            self.initial_mass = mass
        share = 1. if self.pore_fraction is None else self.pore_fraction
        means = np.array([float(u.mean(dtype=xp.float64))*share for u in fields])
        speed2 = sum(u*u for u in fields)
        rms = float(xp.sqrt(speed2.mean()*share))
        peak = float(xp.sqrt(speed2.max()))
        pore = float(fluid.mean()) if self.pore_fraction is None else share
        r = rho[fluid]
        d = dict(iterations=step, superficial_velocity=means.tolist(), velocity_rms=rms,
                 mach_max=peak*np.sqrt(3), rho_min=float(r.min()), rho_max=float(r.max()),
                 density_range=float(r.max()-r.min()), mass=mass,
                 mass_drift=abs(mass-self.initial_mass)/self.initial_mass,
                 reynolds_pore=float(np.linalg.norm(means)/pore*length/nu),
                 reynolds_length_lu=length, reynolds_velocity='norm(superficial_velocity)/porosity')
        passed = False
        if self.previous is not None:
            delta = float(xp.sqrt(sum(((u-v)**2).mean() for u,v in zip(fields,self.previous))*share))
            vec_delta = abs(means-self.previous_means)
            field_limit = self.atol+self.tol*rms
            vector_limit = self.atol+self.tol*np.maximum(abs(means),abs(self.previous_means))
            passed = delta <= field_limit and bool(np.all(vec_delta <= vector_limit))
            d.update(field_residual=delta/max(rms,self.atol), field_change_rms=delta,
                     vector_change=vec_delta.tolist())
        if self.periodic:
            passed = passed and d['mass_drift'] <= self.mass_tol
        if extra:
            d.update(extra)
            passed = passed and extra.get('pressure_checks_passed', True)
        self.streak = self.streak+1 if passed else 0
        d['consecutive_passes'] = self.streak
        self.history.append(d)
        self.previous = tuple(u.copy() for u in fields)
        self.previous_means = means
        return ('converged' if self.streak >= self.required else None), d
