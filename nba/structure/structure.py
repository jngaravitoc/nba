import numpy as np
from scipy.interpolate import BSpline, splrep

class Profiles:
    """
    Compute spherically averaged radial profiles from particle data.

    This class provides utilities to compute density, enclosed mass,
    and potential-related radial profiles from particle positions.
    """

    def __init__(self, pos, edges):
        """
        Initialize the profile calculator.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Cartesian positions of particles.
        edges : array_like
            Radial bin edges.
        """
        self.pos = np.asarray(pos)
        self.edges = np.asarray(edges)

        if self.edges.size <= 1:
            raise ValueError("Edges array must contain more than one element.")
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3).")

    def density(self, smooth=0.0, mass=1.0):
        """
        Compute the spherically averaged density profile.

        Parameters
        ----------
        smooth : float, optional
            Spline smoothing factor passed to ``scipy.interpolate.splrep``.
            If ``smooth <= 0``, no smoothing is applied.
        mass : float or array_like, optional
            Particle mass. If scalar, all particles are assumed to have
            equal mass. If array-like, must have length ``N``.

        Returns
        -------
        r_centers : ndarray
            Radial bin centers.
        dens_profile : ndarray
            Density in each radial bin.
        """

        r = np.linalg.norm(self.pos, axis=1)

        if np.isscalar(mass):
            counts, _ = np.histogram(r, bins=self.edges)
            mass_hist = counts * mass
        else:
            mass_hist, _ = np.histogram(r, bins=self.edges, weights=mass)

        r_outer = self.edges[1:]
        r_inner = self.edges[:-1]
        volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

        dens = mass_hist / volumes
        r_centers = 0.5 * (r_outer + r_inner)

        if smooth > 0:
            mask = dens > 0
            tck = splrep(r_centers[mask], np.log10(dens[mask]), s=smooth)
            dens = 10**(BSpline(*tck)(r_centers))

        return r_centers, dens


    def potential(self, particle_potential):
        """
        Compute a radial profile of summed particle potential per shell.

        Parameters
        ----------
        particle_potential : array_like, shape (N,)
            Per-particle potential values.

        Returns
        -------
        r_centers : ndarray
            Radial bin centers.
        pot_profile : ndarray
            Sum of particle potentials in each radial shell.
        """
        particle_potential = np.asarray(particle_potential)
        if particle_potential.shape[0] != self.pos.shape[0]:
            raise ValueError("particle_potential must have length N.")

        r = np.linalg.norm(self.pos, axis=1)

        pot_profile, _ = np.histogram(
            r, bins=self.edges, weights=particle_potential
        )

        r_centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        return r_centers, pot_profile

    def enclosed_mass(self, mass):
        """
        Compute the enclosed mass profile.

        Parameters
        ----------
        mass : array_like, shape (N,)
            Particle masses.

        Returns
        -------
        r_centers : ndarray
            Radial bin centers.
        mass_enclosed : ndarray
            Enclosed mass within each radius.
        """
        mass = np.asarray(mass)
        if mass.shape[0] != self.pos.shape[0]:
            raise ValueError("mass must have length N.")

        r = np.linalg.norm(self.pos, axis=1)

        shell_mass, _ = np.histogram(
            r, bins=self.edges, weights=mass
        )

        mass_enclosed = np.cumsum(shell_mass)
        r_centers = 0.5 * (self.edges[1:] + self.edges[:-1])

        return r_centers, mass_enclosed

