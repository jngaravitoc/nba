"""
Simulations data reader functionality
"""

import os
import logging
from typing import Union, List, Dict
import numpy as np
import h5py

logger = logging.getLogger(__name__)

class ReadGadgetSim:
    """
    Reader of Gadget-4 HDF5 snapshots.

    Parameters
    ----------
    path : str
        Directory containing the snapshot files.
    snapname : str
        File name of the snapshot, including the ``.hdf5`` extension.

    Examples
    --------
    >>> snap = ReadGadgetSim("/path/to/snapshots", "snapshot_100.hdf5")  # doctest: +SKIP
    >>> header = snap.read_header()  # doctest: +SKIP
    >>> dm = snap.read_snapshot(["pos", "vel", "mass"], ptype="dm")  # doctest: +SKIP
    """

    def __init__(self, path: str, snapname: str):
        self.path = path
        self.snapname = snapname
        self.full_snap_path = os.path.join(self.path, self.snapname)

    def open_snap(self, ptype: str, prop_names: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Read HDF5 datasets of one particle group.

        Lower-level than `read_snapshot`: it takes the names used in the file.

        Parameters
        ----------
        ptype : str
            Particle group in the file, e.g. ``'PartType1'``.
        prop_names : str or list of str
            HDF5 dataset names, e.g. ``'Coordinates'``, ``'Velocities'``.

        Returns
        -------
        dict
            Maps the nba names (``'pos'``, ``'vel'``, ``'mass'``, ``'pot'``,
            ``'pid'``, ``'acc'``) to arrays. Other datasets keep their name.
        """

        prop_map_reverse = {
            'Coordinates': 'pos',
            'Velocities': 'vel',
            'Masses': 'mass',
            'Potential': 'pot',
            'ParticleIDs': 'pid',
            'Acceleration': 'acc'
        }

        if isinstance(prop_names, str):
            prop_names = [prop_names]

        snap_path = f"{self.path}/{self.snapname}"
        data = {}

        with h5py.File(snap_path, 'r') as f:
            if ptype not in f:
                raise ValueError(f"Particle type '{ptype}' not found in snapshot.")

            group = f[ptype]
            for hdf5_key in prop_names:
                if hdf5_key not in group:
                    raise KeyError(f"Property '{hdf5_key}' not found in '{ptype}' group.")
                
                std_key = prop_map_reverse.get(hdf5_key, hdf5_key)  # fallback to raw name if not in map
                data[std_key] = np.array(group[hdf5_key])

        return data


    def read_header(self) -> Dict[str, Union[int, float]]:
        """
        Read the header attributes ``Time``, ``Redshift``, ``BoxSize``,
        ``NumPart_Total``, ``NumPart_Total_HighWord`` and ``MassTable``, when
        present.

        Returns
        -------
        dict
            Header attributes, by name.
        """
        header_keys = ['Time', 'Redshift', 'BoxSize', 'NumPart_Total', 'NumPart_Total_HighWord', 'MassTable']
        metadata = {}

        with h5py.File(self.full_snap_path, 'r') as f:
            header = f['Header'].attrs
            for key in header_keys:
                if key in header:
                    metadata[key] = header[key]
                    logger.debug(f"Header '{key}': {metadata[key]}")

        return metadata

    def has_parttype(self, part_type: str) -> bool:
        """
        Check whether a particle group exists in the file.

        Parameters
        ----------
        part_type : str
            Particle group in the file, e.g. ``'PartType2'``.

        Returns
        -------
        bool
        """
        with h5py.File(self.full_snap_path, 'r') as f:
            return part_type in f

    def read_snapshot(self, quantity: Union[str, List[str]], ptype: str, snapformat=3) -> Union[np.ndarray, Dict[str, np.ndarray]]:       
        """
        Read particle quantities of one particle type.

        Parameters
        ----------
        quantity : str or list of str
            Any of ``'pos'``, ``'vel'``, ``'mass'``, ``'pot'``, ``'pid'`` and
            ``'acc'``.
        ptype : {'dm', 'disk', 'bulge'}
            Particle type: ``PartType1``, ``PartType2`` or ``PartType3``.
        snapformat : int
            Only 3 (Gadget-4 HDF5) is implemented; 1 (Gadget-2/3) and 2
            (ASCII) raise `NotImplementedError`.

        Returns
        -------
        dict
            Maps each quantity to an array, in the file's units and dtype.
        """
        if snapformat == 1:
            raise NotImplementedError("Gadget2/3 not supported yet.")
        elif snapformat == 2:
            raise NotImplementedError("ASCII format not supported yet.")
        elif snapformat == 3:
            prop_map = {
                'pos': 'Coordinates',
                'vel': 'Velocities',
                'mass': 'Masses',
                'pot': 'Potential',
                'pid': 'ParticleIDs',
                'acc': 'Acceleration'
            }

            ptype_map = {
                'dm': 'PartType1',
                'disk': 'PartType2',
                'bulge': 'PartType3'
            }

            if isinstance(quantity, str):
                quantity = [quantity]

            property_name = []
            for i in range(len(quantity)):    
                if quantity[i] not in prop_map:
                    raise ValueError("Invalid quantity. Choose from: ['pos', 'vel', 'mass', 'pot', 'pid', 'acc']")
                else:
                    property_name.append(prop_map[quantity[i]])

            if ptype not in ptype_map:
                raise ValueError("Invalid ptype. Choose from: ['dm', 'disk', 'bulge']")

            part_type = ptype_map[ptype]
            return self.open_snap(part_type, property_name)
        else:
            raise ValueError("Invalid format. Choose from: 1 (Gadget2/3), 2 (ASCII), 3 (Gadget4)")


class ReadGC21:
    """
    Reader of the Garavito-Camargo et al. (2021, GC21) MW-LMC simulations,
    which selects the particles of one halo.

    The dark matter of both halos is in ``PartType1``: the `npart_mw`
    particles with the lowest IDs are the MW, the others the LMC. The MW disk
    and bulge are ``PartType2`` and ``PartType3``.

    Parameters
    ----------
    path : str
        Path to the directory containing the snapshot.
    snapname : str
        File name of the snapshot, including the ``.hdf5`` extension.

    Attributes
    ----------
    full_snap_path : str
        Full path to the snapshot file.
    npart_mw : int
        Number of MW dark matter particles; they have the lowest IDs.
    """

    npart_mw = 100_000_000

    def __init__(self, path: str, snapname: str):
        self.path = path
        self.snapname = snapname
        self.full_snap_path = os.path.join(self.path, self.snapname)

    def read_halo(self, quantity, halo, ptype, randomsample=None, seed=None):
        """
        Read particle quantities of one halo.

        All the dark matter particles are read before the halo is selected, so
        the memory needed is that of the whole snapshot (about 6-7 GB for the
        115M particles of GC21 with positions, velocities and masses).

        Parameters
        ----------
        quantity : str or list of str
            Any of the quantities of `ReadGadgetSim.read_snapshot`, e.g.
            ``['pos', 'vel', 'mass']``. ``'pid'`` is always added.
        halo : {'MW', 'LMC'}
            Halo whose dark matter particles are returned (only used for
            ``ptype='dm'``).
        ptype : {'dm', 'disk', 'bulge'}
            Particle type. The disk and bulge belong to the MW.
        randomsample : int or None
            Number of particles drawn at random, without replacement, from
            the selection (all of them if it is larger).
        seed : int or None
            Seed of the random generator used for ``randomsample``.

        Returns
        -------
        dict
            Maps each quantity, and ``'pid'``, to an array.
        """

        if isinstance(quantity, str):
            quantity = [quantity]
        else:
            quantity = list(quantity)

        if 'pid' not in quantity:
            quantity.append('pid')

        # Read snapshot and header
        GC21 = ReadGadgetSim(self.path, self.snapname)
        #GC21_header = GC21.read_header()
        GC21_dm_data = GC21.read_snapshot(quantity=quantity, ptype=ptype)

        npart_mw = self.npart_mw
        if ptype == 'dm' and halo in ('MW', 'LMC'):
            # The MW particles have the npart_mw lowest IDs: split at the
            # smallest LMC ID, found without sorting all the IDs
            pid = GC21_dm_data['pid']
            if len(pid) > npart_mw:
                cut = np.partition(pid, npart_mw)[npart_mw]
                mask = pid < cut if halo == 'MW' else pid >= cut
            else:
                mask = np.full(len(pid), halo == 'MW')
            for q in quantity:
                GC21_dm_data[q] = GC21_dm_data[q][mask]

        if randomsample:
            npart = len(GC21_dm_data[quantity[0]])
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(npart, min(randomsample, npart), replace=False))
            for q in quantity:
                GC21_dm_data[q] = GC21_dm_data[q][idx]

        return GC21_dm_data

class ReadSheng24:
    """
    Snapshot reader for the Sheng et al. (2024) simulation suite.

    This class provides a thin interface around ``ReadGadgetSim`` with
    additional logic to separate Milky Way (MW) and LMC dark matter
    components based on particle mass.
    """

    def __init__(self, path: str, snapname: str):
        """
        Initialize the Sheng+24 snapshot reader.

        Parameters
        ----------
        path : str
            Path to the directory containing the snapshot.
        snapname : str
            Snapshot filename.
        """
        self.path = path
        self.snapname = snapname
        self.full_snap_path = os.path.join(self.path, self.snapname)

    def get_mw_lmc_ids(self, all_particles_mass):
        """
        Split particle indices into MW and LMC components based on particle mass.

        Assumes exactly two distinct particle masses, assigning the more numerous
        population to the Milky Way (MW) and the less numerous to the LMC.

        Parameters
        ----------
        all_particles_mass : array_like
            Array of particle masses.

        Returns
        -------
        mw_ids : ndarray
            Indices of MW particles.
        lmc_ids : ndarray
            Indices of LMC particles.

        Raises
        ------
        ValueError
            If the number of unique particle masses is not exactly two or if the
            populations have equal size.
        """
        particle_masses = np.unique(all_particles_mass)

        if len(particle_masses) != 2:
            raise ValueError(
                f"Expected exactly 2 unique particle masses, got {len(particle_masses)}"
            )

        ids_1 = np.where(all_particles_mass == particle_masses[0])[0]
        ids_2 = np.where(all_particles_mass == particle_masses[1])[0]

        if len(ids_1) > len(ids_2):
            mw_ids, lmc_ids = ids_1, ids_2
        elif len(ids_2) > len(ids_1):
            mw_ids, lmc_ids = ids_2, ids_1
        else:
            raise ValueError(
                "Particle populations have equal size; cannot distinguish MW/LMC"
            )

        return mw_ids, lmc_ids

    def read_halo(self, quantity, halo, ptype, randomsample=None):
        """
        Read particle data from the snapshot, optionally filtering by halo.

        Parameters
        ----------
        quantity : str or sequence of str
            Particle quantities to read (e.g., ``'pos'``, ``'vel'``, ``'mass'``).
            ``'pid'`` and, for dark matter, ``'mass'`` are always added.
        halo : {'MW', 'LMC'}
            Halo component to select (only applied for ``ptype='dm'``).
        ptype : {'dm', 'disk', 'bulge'}
            Particle type.
        randomsample : None
            Not implemented: any other value raises a ValueError.

        Returns
        -------
        particle_data : dict
            Dictionary mapping quantity names to NumPy arrays.

        Raises
        ------
        ValueError
            If ``randomsample`` is requested.
        """
        if isinstance(quantity, str):
            quantity = [quantity]
        else:
            quantity = list(quantity)

        if 'pid' not in quantity:
            quantity.append('pid')

        # Check if mass is in quantity
        if ptype == 'dm' and 'mass' not in quantity:
            quantity.append('mass')    

        snap = ReadGadgetSim(self.path, self.snapname)
        particle_data = snap.read_snapshot(quantity=quantity, ptype=ptype)

        if ptype == 'dm':
            mw_ids, lmc_ids = self.get_mw_lmc_ids(particle_data['mass'])

            if halo == 'MW':
                ids = mw_ids
            elif halo == 'LMC':
                ids = lmc_ids
            else:
                raise ValueError(f"Unknown halo '{halo}'")

            for q in quantity:
                particle_data[q] = particle_data[q][ids]

        if randomsample is not None:
            raise ValueError("Random sample not implemented yet")

        return particle_data

    def read_header(self):
        """
        Read and return the snapshot header.

        Returns
        -------
        header : dict
            Snapshot header information.
        """
        snap = ReadGadgetSim(self.path, self.snapname)
        return snap.read_header()


