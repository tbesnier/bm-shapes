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
    
class Fractional_BM():
    """
    A fractional brownian motion class constructor
    """
    def __init__(self, H, x0=0):
        """
        Init class
        """
        
        self.H = H ### Hurst exponent
        self.x0 = float(x0)
    
    def davies_harte(self, n_step=50, T=1):
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method

        args:
            T:      length of time (in years)
            N:      number of time steps within timeframe
            H:      Hurst parameter
        '''
        gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
        g = [gamma(k,self.H) for k in range(0,n_step)];    r = g + [0] + g[::-1][0:n_step-1]

        # Step 1 (eigenvalues)
        j = np.arange(0,2*n_step);   k = 2*n_step-1
        lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*n_step))))[::-1]

        # Step 2 (get random variables)
        Vj = np.zeros((2*n_step,2), dtype=np.complex); 
        Vj[0,0] = np.random.standard_normal();  Vj[n_step,0] = np.random.standard_normal()

        for i in range(1,n_step):
            Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
            Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*n_step-i][0] = Vj1;    Vj[2*n_step-i][1] = Vj2

        # Step 3 (compute Z)
        wk = np.zeros(2*n_step, dtype=np.complex)   
        wk[0] = np.sqrt((lk[0]/(2*n_step)))*Vj[0][0];          
        wk[1:n_step] = np.sqrt(lk[1:n_step]/(4*n_step))*((Vj[1:n_step].T[0]) + (complex(0,1)*Vj[1:n_step].T[1]))       
        wk[n_step] = np.sqrt((lk[0]/(2*n_step)))*Vj[n_step][0]       
        wk[n_step+1:2*n_step] = np.sqrt(lk[n_step+1:2*n_step]/(4*n_step))*(np.flip(Vj[1:n_step].T[0]) - (complex(0,1)*np.flip(Vj[1:n_step].T[1])))

        Z = np.fft.fft(wk);     fGn = Z[0:n_step] 
        fBm = np.cumsum(fGn)*(n_step**(-self.H))
        fBm = (T**self.H)*(fBm)
        path = np.array(list(fBm))
        return path.real