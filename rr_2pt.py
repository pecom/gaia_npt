import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.table import Table
import time
import two_pt
import multiprocessing as mp

# full_data = np.loadtxt('./data/funky_data_enc.csv', delimiter=' ')
# coords = full_data[:10000,:3]

coords = np.load('./data/qso_cartesian.npy')
rands = np.load('./data/rand_cartesian.npy')
# coords = coords[:50000, :]

#bins = np.arange(30) # Keep values from 0 to 30 only
bins = np.linspace(0,200, 21, dtype=int)
dlen = len(rands)

Nprocs = 8
full_ndxs = np.arange(dlen)
ndx_lines = np.linspace(0, dlen, Nprocs+1, dtype=int)
ndx_tuple = [(bnd, end) for bnd, end in zip(ndx_lines[:-1], ndx_lines[1:])]
sub_ndx = [full_ndxs[ndx[0]:ndx[1]] for ndx in ndx_tuple]

sub_sum = sum([len(sn) for sn in sub_ndx])

if not(sub_sum == dlen):
    print("MAJOR ISSUE. DATA NOT PARTITIONED PROPERLY")
    sys.exit()

if __name__ == '__main__':
    out_q = mp.Queue()
    procs = []

    t1= time.time()
    for i in range(Nprocs):
        p = mp.Process(target=two_pt.mp_binned_twosamp_2pt, args=(rands, rands, bins, sub_ndx[i], out_q))
        procs.append(p)
        p.start()
    f_xi = []
    for i in range(Nprocs):
        f_xi.append(out_q.get())

    for p in procs:
        p.join()

    t2= time.time()
    # print(f"Done at {t2}")
    xi = np.zeros_like(f_xi[0])
    for fx in f_xi:
        xi += fx

    print(xi[:4])
    print(f"Total time: {t2-t1:0.2f} seconds")

    np.savetxt('./output/xi_rr.csv', xi, delimiter=',')
