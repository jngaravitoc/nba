import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as u
import sys
import pynbody
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

sys.path.append("../scripts/src/")

sys.path.append("/mnt/home/ecunningham/python")
#plt.style.use('~/matplotlib.mplstyle')
import gizmo_analysis as ga
import halo_analysis as halo
import nba

# 
import pynbody_routines as pr 
import io_gizmo_pynbody as fa
import plotting as pl

from scipy.linalg import norm
import h5py
import itertools


def sim_angmom(sim, snap):
    #sim='m12b'
    sim_directory = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/".format(sim)
    snap_times = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt".format(sim)
    times = np.loadtxt(snap_times, usecols=3)
    m12i = fa.FIRE(sim, remove_satellite=True)
    sub_not_sat = m12i.subhalos(snap)
    mcut = np.where(np.log10(sub_not_sat.star['mass']) > 9)
    halo_kin = nba.kinematics.Kinematics(sub_not_sat.star['pos'][mcut], sub_not_sat.star['vel'][mcut])
    L  = halo_kin.part_angular_momentum()[:3]
    pos_r = L / norm(L, axis=0)
    x = np.where(np.abs(pos_r[0]) >= 0)
    Nsats = len(x[0])
    print("Nsats=", len(x[0]))
    #x = np.isnan(pos_r[0])
    pos_rx = np.zeros((3, Nsats))
    pos_rx[0] = pos_r[0][x]
    pos_rx[1] = pos_r[1][x]
    pos_rx[2] = pos_r[2][x]
    
    return pos_rx


def delta_k(pos_rx, k):
    npoints = np.shape(pos_rx)[1]
    index = np.linspace(0, npoints-1, npoints)
    index_comb = np.array(list(itertools.combinations(index, k))).astype(int)
    
    x3 = np.zeros((len(index_comb), k))
    y3 = np.zeros((len(index_comb), k))
    z3 = np.zeros((len(index_comb), k))
    for i in range(len(x3)):
        x3[i] = pos_rx.T[index_comb[i],0]
        y3[i] = pos_rx.T[index_comb[i],1]
        z3[i] = pos_rx.T[index_comb[i],2]

    #print(np.shape(x3), len(x3))
    delta_sph = np.zeros(len(x3))
    for n in range(len(x3)):
        x_mean = np.mean(x3[n])
        y_mean = np.mean(y3[n])
        z_mean = np.mean(z3[n])

        delta_sph_k = np.zeros(k)
        #print(len(lorb3))
        mean_vec = np.array([x_mean, y_mean, z_mean])/norm(np.array([x_mean, y_mean, z_mean]))
        for p in range(k):
            all_vec =  np.array([x3[n, p], y3[n, p], z3[n, p]])/norm(np.array([x3[n, p], y3[n, p], z3[n, p]]))
            delta_sph_k[p] = np.arccos((np.dot(mean_vec, all_vec)))*180/np.pi
            #print(delta_sph_k)
        delta_sph[n] = np.sqrt(np.sum(delta_sph_k**2, axis=0)/k)
    return np.min(delta_sph)

if __name__ == "__main__":
    dsph_m12b = np.zeros((21, 7))
    i=0
    for sn in range(370, 461, 5):
        pos_sn = sim_angmom('m12b', sn)
        for k in range(3, 9):
            dsph_m12b[i, k-3] = delta_k(pos_sn, k)
            print('-> Done k: {}'.format(k))
        i+=1

np.savetxt('delta_sph_m12b_mcut9_k9.txt', dsph_m12b)
