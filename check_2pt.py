import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.table import Table
import time
import two_pt

full_data = np.loadtxt('./data/funky_data_enc.csv', delimiter=' ')
coords = full_data[:10000,:3]

bins = np.arange(30) # Keep values from 0 to 30 only
# bins = np.linspace(0,200, 21, dtype=int)
dlen = len(coords)

if __name__ == '__main__':
    t1= time.time()
    xi = two_pt.binned_twosamp_2pt(coords, coords, bins)
    t2= time.time()
    # print(f"Done at {t2}")

    print(xi[:4])
    print(f"Total time: {t2-t1:0.2f} seconds")

    np.savetxt('./output/check_xi.csv', xi, delimiter=',')
