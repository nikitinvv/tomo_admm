import cupy as cp
import dxchange
import numpy as np
import tomo_admm as pt

if __name__ == "__main__":

    # Model parameters
    n = 256  # object size in x,y
    nz = 256  # object size in z
    ntheta = 64  # number of angles (rotations)
    center = n/2  # rotation center
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')  # angles

    # Load object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = dxchange.read_tiff('data/delta-chip-256.tiff')
    u0 = cp.array(delta+1j*beta)
    
    niter = 64
    with pt.SolverTomo(ntheta, nz, n, center) as slv:
        # generate data
        data = slv.fwd_tomo(u0, theta)
        # initial guess
        u = cp.zeros([nz, n, n], dtype='complex64')
        u = slv.cg_tomo(data, u, theta, niter, dbg=True)
        # save results
        dxchange.write_tiff(u.real.get(),  'rec/delta', overwrite=True)
        dxchange.write_tiff(u.imag.get(),  'rec/beta', overwrite=True)
        dxchange.write_tiff(data.real.get(),  'datar/r', overwrite=True)
        dxchange.write_tiff(data.imag.get(),  'datar/i', overwrite=True)
        