"""
Simulations data reader
"""

import os
import logging
from typing import Union, List, Dict
import numpy as np
import h5py

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ReadGadgetSim:
    """
    A class to read Gadget4 (HDF5) simulation snapshots.

    Parameters
    ----------
    path : str
        Directory containing the snapshot files.
    snapname : str
        Base name of the snapshot file (without .hdf5).
    """

    def __init__(self, path: str, snapname: str):
        self.path = path
        self.snapname = snapname
        self.full_snap_path = os.path.join(self.path, self.snapname)

    def open_snap(self, ptype: str, prop_names: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Read specified properties from the HDF5 snapshot file for a given particle type.

        Parameters
        ----------
        ptype : str
            Particle type group in the file (e.g., 'PartType1', 'PartType2', ...)
        prop_names : str or list of str
            HDF5 dataset property names to load (e.g., 'Coordinates', 'Velocities')

        Returns
        -------
        dict
            Dictionary mapping standard keys ('pos', 'vel', 'mass', etc.) to NumPy arrays
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
        Read selected header attributes.

        Returns
        -------
        dict
            Header information.
        """
        header_keys = ['Time', 'Redshift', 'BoxSize', 'NumPart_Total', 'NumPart_Total_HighWord', 'MassTable']
        metadata = {}

        with h5py.File(self.full_snap_path, 'r') as f:
            header = f['Header'].attrs
            for key in header_keys:
                if key in header:
                    metadata[key] = header[key]
                    logger.info(f"Header '{key}': {metadata[key]}")

        return metadata

    def has_parttype(self, part_type: str) -> bool:
        """
        Check if a particle type exists in the file.

        Parameters
        ----------
        part_type : str

        Returns
        -------
        bool
        """
        with h5py.File(self.full_snap_path, 'r') as f:
            return part_type in f

    def read_snapshot(self, quantity: Union[str, List[str]], ptype: str, snapformat=3) -> Union[np.ndarray, Dict[str, np.ndarray]]:       
        """
        Load particle data depending on the snapshot format.

        Parameters
        ----------
        snapformat : int
            1: Gadget2/3 (not implemented), 2: ASCII (not implemented), 3: Gadget4 (HDF5)
        quantity : str
            'pos', 'vel', 'mass', 'pot', 'pid', 'acc'
        ptype : str
            'dm', 'disk', 'bulge'

        Returns
        -------
        np.ndarray
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
    Class to read GC21-format Gadget4 snapshots and extract halo-specific data.

    Parameters
    ----------
    path : str
        Path to the directory containing the snapshot.
    snapname : str
        Name of the snapshot file (without .hdf5 extension).

    Attributes
    ----------
    full_snap_path : str
        Full path to the snapshot file.
    """

    def __init__(self, path: str, snapname: str):
        self.path = path
        self.snapname = snapname
        self.full_snap_path = os.path.join(self.path, self.snapname)

    def read_halo(self, quantity, halo):
        """
        Load particle data for a specified halo ("MW" or "LMC") and desired quantities.

        Parameters
        ----------
        quantity : str or list of str
            Particle properties to read, e.g., ['pos', 'vel']. Will automatically include 'pid' for sorting.
        halo : str
            Which halo to return particles from: 'MW' or 'LMC'.

        Returns
        -------
        dict of numpy.ndarray
            Dictionary with keys matching `quantity`, containing filtered arrays for the selected halo.

        Raises
        ------
        ValueError
            If an unknown halo is specified.
        """

        if isinstance(quantity, str):
            quantity = [quantity]
        else:
            quantity = list(quantity)

        if 'pid' not in quantity:
            quantity.append('pid')

        # Read snapshot and header
        GC21 = ReadGadgetSim(self.path, self.snapname)
        GC21_dm_data = GC21.read_snapshot(quantity=quantity, ptype='dm')
        GC21_header = GC21.read_header()

        # NOTE: This should be parameterized ideally
        npart_mw = 100_000_000
        npart_sat = GC21_header['NumPart_Total'][1] - npart_mw

        #ids_sort = np.argsort(GC21_dm_data['pid'])

        #if halo == 'MW':
        #    halo_idx = ids_sort[:npart_mw]
        #elif halo == 'LMC':
        #    halo_idx = ids_sort[npart_mw:]
        #    assert len(halo_idx) == npart_sat
        #else:
        #    raise ValueError("Halo must be 'MW' or 'LMC'.")

        #for q in quantity:
        #    GC21_dm_data[q] = GC21_dm_data[q][halo_idx]
        
        if halo == 'MW':
            halo_ids = np.sort(GC21_dm_data['pid'])[:npart_mw]
        elif halo == 'LMC':
            halo_ids = np.sort(GC21_dm_data['pid'])[npart_mw:]

        # Build a boolean mask from IDs
        mask = np.isin(GC21_dm_data['pid'], halo_ids)
        for q in quantity:
            GC21_dm_data[q] = GC21_dm_data[q][mask]
        return GC21_dm_data
