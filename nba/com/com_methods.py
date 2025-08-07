#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

@njit
def ssphere_numba(xyz, vxyz, mass, delta=0.025, rcut=20.0):
    """
    Shrinking Sphere method with Numba acceleration.
    
    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Particle positions.
    vxyz : np.ndarray, shape (N, 3)
        Particle velocities.
    mass : np.ndarray, shape (N,)
        Particle masses.
    delta : float
        Convergence criterion.
    rcut : float
        Radius to compute final COM velocity.

    Returns
    -------
    com_pos : np.ndarray, shape (3,)
        Center of mass position.
    com_vel : np.ndarray, shape (3,)
        Center of mass velocity (within rcut).
    """
    N_init = xyz.shape[0]
    indices = np.arange(N_init)
    
    # Initial COM
    msum = np.sum(mass)
    com_pos = np.zeros(3)
    for i in range(3):
        com_pos[i] = np.sum(xyz[:, i] * mass) / msum

    shift = 500
    while shift > delta:
        # Compute squared distance from COM
        r2 = np.zeros(len(indices))
        for i in range(len(indices)):
            dx = xyz[indices[i], 0] - com_pos[0]
            dy = xyz[indices[i], 1] - com_pos[1]
            dz = xyz[indices[i], 2] - com_pos[2]
            r2[i] = dx*dx + dy*dy + dz*dz

        r2_max = np.max(r2)
        r2_cut = r2_max * 0.975**2

        # Select indices within cut radius
        count = 0
        for i in range(len(r2)):
            if r2[i] < r2_cut:
                count += 1

        if count < max(1000, int(0.01 * N_init)):
            break

        new_indices = np.empty(count, dtype=np.int64)
        j = 0
        for i in range(len(r2)):
            if r2[i] < r2_cut:
                new_indices[j] = indices[i]
                j += 1

        indices = new_indices

        msum = 0.0
        new_com_pos = np.zeros(3)
        for i in range(len(indices)):
            m = mass[indices[i]]
            msum += m
            for j in range(3):
                new_com_pos[j] += xyz[indices[i], j] * m
        for j in range(3):
            new_com_pos[j] /= msum

        shift = np.sqrt(np.sum((new_com_pos - com_pos)**2))
        com_pos = new_com_pos

    # Final velocity COM inside rcut sphere
    r_cut2 = rcut**2
    msum = 0.0
    com_vel = np.zeros(3)
    for i in range(len(indices)):
        dx = xyz[indices[i], 0] - com_pos[0]
        dy = xyz[indices[i], 1] - com_pos[1]
        dz = xyz[indices[i], 2] - com_pos[2]
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < r_cut2:
            m = mass[indices[i]]
            msum += m
            for j in range(3):
                com_vel[j] += vxyz[indices[i], j] * m
    if msum > 0:
        for j in range(3):
            com_vel[j] /= msum

    return com_pos, com_vel



class CenterHalo:
    def __init__(self, Halo):
        self.pos = Halo['pos']
        self.vel = Halo['vel']
        self.mass = Halo['mass']
        self.pot = Halo.get('pot', None)  # Optional

    def recenter(self, vec: np.ndarray, cm: np.ndarray) -> np.ndarray:
        """Subtract center-of-mass vector from a 2D array of vectors."""
        return vec - cm

    def min_potential(self, disk_pot=None, rcut: float = 2.0):
        """Center-of-mass position and velocity near the potential minimum."""
        if hasattr(self, "pot"):
            disk_pot = self.pot
        
        min_idx = np.argmin(disk_pot)
        center = self.pos[min_idx]

        # Distance to potential minimum
        r = np.linalg.norm(self.pos - center, axis=1)
        idx = np.where(r < rcut)[0]

        x_cm = np.mean(self.pos[idx, 0])
        y_cm = np.mean(self.pos[idx, 1])
        z_cm = np.mean(self.pos[idx, 2])
        vx_cm = np.mean(self.vel[idx, 0])
        vy_cm = np.mean(self.vel[idx, 1])
        vz_cm = np.mean(self.vel[idx, 2])

        return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

    def velocities_com(self, cm_pos: np.ndarray, r_cut: float = 20.0) -> np.ndarray:
        """Compute COM velocity within radius `r_cut` of `cm_pos`."""
        dist = np.linalg.norm(self.pos - cm_pos, axis=1)
        mask = dist < r_cut
        return np.mean(self.vel[mask], axis=0)

    def mean_pos(self, rmin: float = 0, rmax: float = 0):
        if rmin < 0 or rmax < 0:
            raise ValueError("rmin and rmax must be non-negative")
        if rmin > rmax:
            raise ValueError("rmin must be less than or equal to rmax")

        if rmin == 0 and rmax == 0:
            weights = self.mass
            pos = self.pos
            vel = self.vel
        else:
            r = np.linalg.norm(self.pos, axis=1)
            mask = (r > rmin) & (r < rmax)
            if not np.any(mask):
                raise ValueError("No particles found in the rmin–rmax range")
            weights = self.mass[mask]
            pos = self.pos[mask]
            vel = self.vel[mask]

        total_mass = np.sum(weights)
        if total_mass == 0:
            raise ValueError("Total mass is zero — cannot compute center of mass")

        com_pos = np.sum(pos * weights[:, None], axis=0) / total_mass
        com_vel = np.sum(vel * weights[:, None], axis=0) / total_mass
        return com_pos, com_vel

    def mean_pos_from_arrays(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray):
        """Helper to compute COM from given arrays."""
        total_mass = np.sum(mass)
        if total_mass == 0:
            raise ValueError("Total mass is zero — cannot compute center of mass")
        com_pos = np.sum(pos * mass[:, None], axis=0) / total_mass
        com_vel = np.sum(vel * mass[:, None], axis=0) / total_mass
        return com_pos, com_vel
    
    def shrinking_sphere(self, delta=0.025, minNpart=1000, rcut_vel = 20):
        xyz = self.pos
        vxyz = self.vel
        mass = self.mass

        indices = np.arange(len(mass))
        N_init = len(indices)
        shift = 500

        com_pos, _ = self.mean_pos()
        
        while shift > delta:
            r2 = np.sum((xyz[indices] - com_pos)**2, axis=1)
            rmax2 = np.max(r2)
            mask = r2 < (rmax2 * 0.975**2)
            indices = indices[mask]

            if len(indices) < max(minNpart, 0.01 * N_init):
                break

            new_com_pos = np.sum(xyz[indices] * mass[indices, None], axis=0) / np.sum(mass[indices])
            shift = np.linalg.norm(new_com_pos - com_pos)
            com_pos = new_com_pos

        # Final COM velocity using selected particles
        #r_cut = 20  # kpc, or make this a parameter
        r = np.linalg.norm(xyz[indices] - com_pos, axis=1)
        inside = r < rcut_vel
        final_vel = np.sum(vxyz[indices][inside] * mass[indices][inside, None], axis=0) / np.sum(mass[indices][inside])

        return com_pos, final_vel

    def shrinking_sphere_numba(self, delta=0.025, rcut=20.0):
        return ssphere_numba(self.pos, self.vel, self.mass, delta, rcut)

   