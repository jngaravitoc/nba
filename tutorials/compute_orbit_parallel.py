import numpy as np
from nba.orbits import orbit
import schwimmbad

def worker(task):
    init_snap = task
    print(init_snap)
    pos_com, vel_com = orbit(snapshot+out_name, init_snap, init_snap, 0, 0, [nhost, nsat], snap_format, com_method2)
    pos_com_sat, vel_com_sat = orbit(snapshot+out_name, init_snap, init_snap, 1, 1, [nhost, nsat], snap_format, com_method1)
    return pos_com, vel_com, pos_com_sat, vel_com_sat
    
def main(pool):
    nsnaps = np.arange(init_snap, final_snap+1, 1, dtype=int)
    #asks = list(zip(nsnaps))
    results = pool.map(worker, nsnaps)
    pool.close()
    return results 


if __name__ == "__main__":
    snapshot = "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5/out/"
    out_name = 'MWLMC5_100M_b0_vir_OM3_G4'
    init_snap = 0 
    final_snap = 400
    snap_format = 3 # gadget4 - hdf5
    com_method1 = 'shrinking'
    com_method2 = 'diskpot'
    nhost=100000000
    nsat=15000000
    orbit_name = "orbit_mwlmc5.txt"

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Schwimmbad")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    results = main(pool)
 
    #pos_com_host, vel_com_host = orbit(snapshot+out_name, init_snap, final_snap, 0, 0, [nhost, nsat], snap_format, com_method2)
    # Save dat
    print(np.shape(list(results)))
    print(np.shape(np.array(results)[:,0,0,:]))
    np.savetxt("test_parallel_orbit_pos.txt", np.array(results)[:,0].reshape(11, 3))
    np.savetxt("test_parallel_orbit_vel.txt", np.array(results)[:,1].reshape(11, 3))
    np.savetxt("test_parallel_orbit_sat_pos.txt", np.array(results)[:,2].reshape(11, 3))
    np.savetxt("test_parallel_orbit_sat_vel.txt", np.array(results)[:,3].reshape(11, 3))

