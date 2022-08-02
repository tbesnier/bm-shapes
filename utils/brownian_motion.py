import numpy as np
import six

class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self, n_step = 50, T = 1):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        
        h = T /n_step
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal(0, h)
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
class Bridge():
    """
    A Brownian bridge class constructor
    """
    def __init__(self, x0=0 , x1=0):
        """
        Init class
        """
        #assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        #assert (type(x1)==float or type(x1)==int or x1 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
        self.x1 = float(x1)
    
    def gen_traj(self, eta = 1, n_step = 50, T = 1.0):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        
        dt = T / (n_step-1)                                                        
        dt_sqrt = np.sqrt(dt)
        B = np.empty(n_step, dtype=np.float32)
        B[0] = self.x0
        for n in six.moves.range(n_step - 2):                                          
            t = n * dt
            xi = (np.random.randn(1) * dt_sqrt * eta)
            B[n + 1] = B[n] + ((self.x1 - B[n])/ (T - t))*dt + xi
        B[-1] = self.x1  
        
        return B
    
class Diffusion_process():
    """
    A Diffusion process class constructor
    """
    def __init__(self, b, sigma, x0=0):
        """
        Init class
        """
        #assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        #assert (type(x1)==float or type(x1)==int or x1 is None), "Expect a float or None for the initial value"
        
        self.b = b ### drift term
        self.sigma = sigma ### diffusion term
        self.x0 = float(x0)
    
    def gen_traj(self, eta=1, n_step = 50, T = 1.0):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        
        dt = T / (n_step-1)                                                        
        dt_sqrt = np.sqrt(dt)
        B = np.empty(n_step, dtype=np.float32)
        B[0] = self.x0
        for n in six.moves.range(n_step - 1):                                          
            t = n * dt
            xi = (np.random.randn(1) * dt_sqrt * eta)
            B[n + 1] = B[n] + self.b(t, B[n])*dt + self.sigma(t, B[n])*xi
        
        return B