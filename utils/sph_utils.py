from scipy.special import sph_harm
import numpy as np
import scipy.integrate as integrate
from utils import brownian_motion

class SphHarmBasis():
    def __init__(self, n_coeffs=16):
        self._n_coeffs = n_coeffs
        self.basis = self.sph_harm_basis()
    
    def get_sph_harm_function(self, l, m):
        """Compute real spherical harmonic basis function"""
        def basis_function(theta, phi):
            Y = sph_harm(abs(m), l, phi, theta)
            
            if m < 0:
                Y = np.sqrt(2) * (-1)**m * Y.imag
            elif m > 0:
                Y = np.sqrt(2) * (-1)**m * Y.real
                
            return Y.real
        
        return basis_function

    def sph_harm_basis(self):
        """Get a specified number of basis functions"""
        basis_functions = []

        dimension = 0
        l, m = 0, 0

        while dimension < self._n_coeffs:
            while m <= l:
                basis_functions.append(self.get_sph_harm_function(l, m))
                m += 1
                dimension += 1
            
            l += 1
            m = -l
        
        return basis_functions

    def sph_harm_coeff(self, Y, f):
        """Compute spherical harmonic coefficients"""
        def integrand(phi, theta):
            return f(theta, phi) * Y(theta, phi) * np.sin(theta)
        options={'limit':25}
        res = integrate.nquad(integrand, [[0., 2*np.pi], [0., np.pi]])[0]
        return res

    def sph_harm_transform(self, f, basis=None):
        """Get spherical harmonic coefficients for a function in a basis"""
        if basis is None:
            basis = self.basis
        
        coeffs = []

        for Y in basis:
            coeffs.append(self.sph_harm_coeff(Y, f))
        return coeffs

    def sph_harm_reconstruct(self, coeffs, basis=None):
        """Reconstruct a function from basis and corresponding coefficients"""
        if basis is None:
            basis = self.basis
        
        return lambda theta, phi : np.dot(coeffs, [f(theta, phi) for f in basis])
    
    def sph_harm_reconstruct_random(self, coeffs, Q, t, theta, phi, n_step = 2, basis=None):
        """Reconstruct a function from basis and corresponding coefficients"""
        if basis is None:
            basis = self.basis
        
        b = np.array([brownian_motion.Brownian(0).gen_normal(n_step = n_step, T = t) for coeff in coeffs])
        coeffs_stoch = np.array([coeffs for i in range(n_step)]).T + Q.dot(b)
        
        paths = []
        for it in range(coeffs_stoch.shape[1]):
            
            paths.append(np.dot(coeffs_stoch[:,it], [f(theta, phi) for f in basis]))
            
        return(np.array(paths))
    
    def sph_harm_reconstruct_bridge(self, coeffs_source, coeffs_target, t, theta, phi, Q, n_step = 2, basis=None):
        """Reconstruct a function from basis and corresponding coefficients"""
        if basis is None:
            basis = self.basis
            
        vec = np.vstack((coeffs_source, coeffs_target))
        
        b = np.array([brownian_motion.Bridge(vec[0,i], vec[1,i]).gen_traj(eta = Q[i,i], n_step = n_step, T = t) for i in range(len(coeffs_source))])
        coeffs_stoch = b
        
        paths = []
        for it in range(coeffs_stoch.shape[1]):
            
            paths.append(np.dot(coeffs_stoch[:,it], [f(theta, phi) for f in basis]))
            
        return(np.array(paths), b)
    
    def sph_harm_reconstruct_sde(self, coeffs_source, Q, t, theta, phi, b, sigma, n_step = 2, basis=None):
        """Reconstruct a function from basis and corresponding coefficients"""
        if basis is None:
            basis = self.basis
        
        s = np.array([brownian_motion.Diffusion_process(b, sigma, coeffs_source[i]).gen_traj(eta = Q[i,i], n_step = n_step, T = t) for i in range(len(coeffs_source))])
        coeffs_stoch = s
        
        paths = []
        for it in range(coeffs_stoch.shape[1]):
            
            paths.append(np.dot(coeffs_stoch[:,it], [f(theta, phi) for f in basis]))
            
        return(np.array(paths), s)
    
    def sph_harm_reconstruct_fractional(self, coeffs, Q, t, theta, phi, H, n_step = 2, basis=None):
        """Reconstruct a function from basis and corresponding coefficients"""
        if basis is None:
            basis = self.basis
        
        b = np.array([brownian_motion.Fractional_BM(H,0).davies_harte(n_step = n_step, T = t) for coeff in coeffs])
        coeffs_stoch = np.array([coeffs for i in range(n_step)]).T + Q.dot(b)
        
        paths = []
        for it in range(coeffs_stoch.shape[1]):
            
            paths.append(np.dot(coeffs_stoch[:,it], [f(theta, phi) for f in basis]))
            
        return(np.array(paths))