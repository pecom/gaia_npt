import numpy as np
import healpy as hp
from healpy.newvisufunc import projview
from astropy.table import Table
import scipy.stats as stats

import matplotlib
from matplotlib import pyplot as plt

import pyccl as ccl
from astropy import units as u
from astropy.coordinates import SkyCoord
import treecorr
import pickle 
import os

rng = np.random.default_rng()
pdir = os.getenv('PSCRATCH')
hdir = os.getenv('HOME')


h = .7
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                            h=h, n_s=0.95, sigma8=0.8,
                            transfer_function='bbks')
gdist = lambda zs: cosmo.comoving_radial_distance(1/(1+zs)) * 1/h


def gen_vector(r, ra, dec):
    # Convert a radius, ra, and dec into a cartesian vector
    theta = np.radians(-dec+90.)
    phi = np.radians(360.-ra)
    x = r*np.sin(theta)*np.cos(phi).data
    y = r*np.sin(theta)*np.sin(phi).data
    z = r*np.cos(theta).data
    return x,y,z

def get_fkp(zs, bins=10):
    # Calculate n(z) and then the FKP weight
    bin_nz, bin_edges, binnumber = stats.binned_statistic(zs, zs, statistic='count', bins=bins)
    bin_mpchinv = gdist(bin_edges)
    bin_vols = np.diff(np.pi*4/3*(bin_mpchinv**3))
    units_nz = bin_nz[0]/bin_vols
    gal_nz = units_nz[binnumber-1]    
    fkp_weights = 1/(1+gal_nz*1e4)
    return fkp_weights

def read_quaia(ddir = hdir):
    qfits = Table.read(ddir + '/quaia/data/quaia_G20.0.fits')
    qso_red = qfits['redshift_quaia']
    qso_ra = qfits['ra']
    qso_dec = qfits['dec']
    return qso_ra, qso_dec, qso_red

def read_random(qso_red, ddir = hdir, N_r=0):
    qrand = Table.read(ddir + '/quaia/data/random_G20.0_10x.fits')
    if N_r==0:
        N_r = len(qrand)//10
    
    rand_ndx = rng.choice(len(qrand), N_r, replace=False)
    rand_red = rng.choice(qso_red, N_r)
    rand_ra = qrand['ra'][rand_ndx]
    rand_dec = qrand['dec'][rand_ndx]
    
    return rand_ra, rand_dec, rand_red

def fin_format(ra, dec, zs, fkp=True, fkp_bins=25):
    dist = gdist(zs)
    x,y,z = gen_vector(dist, ra, dec)
    if fkp:
        weights = get_fkp(zs, bins=fkp_bins)
    else:
        weights = np.ones(len(ra))
    cartesian_weighted = np.vstack((x,y,z,weights)).T
    return cartesian_weighted


def save_format(cart_w, fname, odir=hdir, npy=True):
    if npy:
        np.save(f'{odir}/output/{fname}_py.npy', cart_w)
    np.savetxt(f'{odir}/output/{fname}_enc.csv', cart_w)
    print("All saved")