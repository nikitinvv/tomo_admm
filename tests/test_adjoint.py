import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np
import tomo_admm as pt

if __name__ == "__main__":

    # Model parameters
    n = 256  # object size in x,y
    nz = 256  # object size in z
    ntheta = 128  # number of angles (rotations)
    center = n/2  # rotation center
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')  # angles
    niter = 64  # tomography iterations
    # Load object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = dxchange.read_tiff('data/delta-chip-256.tiff')
    u0 = cp.array(delta+1j*beta)
    # Class gpu solver
    with pt.SolverTomo(ntheta, nz, n, center) as slv:
        # generate data
        data = slv.fwd_tomo(u0,theta)
        dxchange.write_tiff(data.real.get(),'rec/r',overwrite=True)        
        dxchange.write_tiff(data.imag.get(),'rec/i',overwrite=True)        
        # # adjoint test
        u1 = slv.adj_tomo(data,theta)
        data1 = slv.fwd_tomo(u1,theta)

        t1 = np.sum(data*np.conj(data))
        t2 = np.sum(u0*np.conj(u1))
        print(f"Adjoint test: {t1.real:06f}{t1.imag:+06f}j "
              f"=? {t2.real:06f}{t2.imag:+06f}j")
        
        print(np.sum(data*np.conj(data1))/np.sum(data1*np.conj(data1)))    
        print(np.sum(u0*np.conj(u1))/np.sum(u1*np.conj(u1)))   
        # np.testing.assert_allclose(t1, t2, atol=1e-3)


        data = slv.fwd_reg(u0)
        # dxchange.write_tiff(data.real.get(),'rec/r',overwrite=True)        
        # dxchange.write_tiff(data.imag.get(),'rec/i',overwrite=True)        
        # # adjoint test
        u1 = slv.adj_reg(data)
        data1 = slv.fwd_reg(u1)

        t1 = np.sum(data*np.conj(data))
        t2 = np.sum(u0*np.conj(u1))
        print(f"Adjoint test: {t1.real:06f}{t1.imag:+06f}j "
              f"=? {t2.real:06f}{t2.imag:+06f}j")
        
        print(np.sum(data*np.conj(data1))/np.sum(data1*np.conj(data1)))    
        print(np.sum(u0*np.conj(u1))/np.sum(u1*np.conj(u1)))   
        # np.testing.assert_allclose(t1, t2, atol=1e-3)
