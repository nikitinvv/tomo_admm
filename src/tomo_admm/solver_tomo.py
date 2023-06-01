"""Module for tomography."""

import cupy as cp
import numpy as np
from tomo_admm.radonusfft import radonusfft


class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    """

    def __init__(self, ntheta, nz, n, center):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, nz, n, center, 1)
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, theta):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        u0 = u.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u0.data.ptr, theta.data.ptr, 0)   
        # normalization
        res/=cp.sqrt((self.ntheta * self.n)/2)
        return res

    def adj_tomo(self, data, theta):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64')
        data0 = data.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data0.data.ptr, theta.data.ptr, 0)
        # normalization
        res/=cp.sqrt((self.ntheta * self.n)/2)
        return res

    def line_search(self, minf, gamma, Ru, Rd):
        """Line search for the step sizes gamma"""
        while(minf(Ru)-minf(Ru+gamma*Rd) < 0):
            gamma *= 0.5
        return gamma
    
    def cg_tomo(self, xi0, u, theta, titer, dbg=False):
        """CG solver for ||Ru-xi0||_2^2"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u, theta)            
            grad = self.adj_tomo(Ru-xi0, theta)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d,theta)
            gamma = 0.5*self.line_search(minf, 2, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if dbg:
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u
    
        
    ################# ADMM
    def line_search_ext(self, minf, gamma, Ru, Rd, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Ru, gu)-minf(Ru+gamma*Rd, gu+gamma*gd) < 0):
            gamma *= 0.5
        return gamma            

    def fwd_reg(self, u):
        """Forward operator for regularization (J)"""
        res = cp.get_array_module(u).zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    def adj_reg(self, gr):
        """Adjoint operator for regularization (J*)"""
        res = cp.get_array_module(gr).zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        return -res
        
    def solve_reg(self, u, mu, rho, alpha):
        """Solution of the L1 problem by soft-thresholding"""
        z = self.fwd_reg(u)+mu/rho
        za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/rho] = 0
        z[:, za > alpha/rho] -= alpha/rho * \
            z[:, za > alpha/rho]/(za[za > alpha/rho])
        return z        
        
    def cg_tomo_ext(self, xi0, xi1, u, theta, rho, titer, dbg=False):
        """CG solver for ||Ru-xi0||_2^2 + rho||Ju-xi1||_2^2"""
        # minimization functional
        def minf(Ru,Ju):
            f = cp.linalg.norm(Ru-xi0)**2+rho*cp.linalg.norm(Ju-xi1)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u,theta)            
            Ju = self.fwd_reg(u)            
            grad = self.adj_tomo(Ru-xi0,theta) + rho*self.adj_reg(Ju-xi1)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d,theta)
            Jd = self.fwd_reg(d)
            gamma = 0.5*self.line_search_ext(minf, 1, Ru, Rd,Ju,Jd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if dbg:
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru,Ju)))
        return u
    
    def admm(self, data, h, psi, lamd, u, theta, alpha, titer, niter, dbg=False):
        """ ADMM for laminography problem with TV regularization"""
        rho = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0 = h.copy()
            # laminography problem
            u = self.cg_tomo_ext(data, psi-lamd/rho, u, theta, rho, titer, False)            
            # regularizer problem
            psi = self.solve_reg(u, lamd, rho, alpha)
            # h updates
            h = self.fwd_reg(u)
            # lambda update
            ##Slow version:
            lamd = lamd + rho * (h-psi)
            # update rho for a faster convergence
            rho = self.update_penalty(psi, h, h0, rho)            
            # Lagrangians difference between two iterations
            if dbg:
                lagr = self.take_lagr(
                    u, psi, data, h, lamd, theta, alpha,rho)
                print("%d/%d) rho=%.2e, Lagrangian terms:   %.2e %.2e %.2e %.2e, Sum: %.2e" %
                        (m, niter, rho, *lagr))
        return u

    def take_lagr(self, u, psi, data, h, lamd, theta, alpha, rho):
        """ Lagrangian terms for monitoring convergence"""
        lagr = np.zeros(5, dtype="float32")
        Lu = self.fwd_tomo(u, theta)
        lagr[0] += np.linalg.norm(Lu-data)**2
        lagr[1] = alpha*np.sum(np.sqrt(np.real(np.sum(psi*np.conj(psi), 0))))        
        lagr[2] = np.sum(np.real(np.conj(lamd)*(h-psi)))        
        lagr[3] = rho*np.linalg.norm(h-psi)**2
        lagr[4] = np.sum(lagr[:4])
        return lagr
    
    def update_penalty(self, psi, h, h0, rho):
        """Update rhofor a faster convergence"""
        r = np.linalg.norm(psi - h)**2
        s = np.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho