import numpy as np
import matplotlib.pyplot as plt
import pywt


def imageplot(f, str='', sbpt=[]):
    """
        Use nearest neighbor interpolation for the display.
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    plt.axis('off')
    if str != '':
        plt.title(str)
        
def rescale(f,a=0,b=1):
    """
        Rescale linearly the dynamic of a vector to fit within a range [a,b]
    """
    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g / v
    return a + g*(b-a)

def plot_wavelet_o(fW, Jmin=0):
    """
        plot_wavelet - plot wavelets coefficients.

        U = plot_wavelet(fW, Jmin):

        Copyright (c) 2014 Gabriel Peyre
    """
    def rescaleWav(A):
        v = abs(A).max()
        B = A.copy()
        if v > 0:
            B = .5 + .5 * A / v
        return B
    ##
    n = fW.shape[1]
    Jmax = int(np.log2(n)) - 1
    U = fW.copy()
    for j in np.arange(Jmax, Jmin - 1, -1):
        U[:2 ** j:,    2 ** j:2 **
            (j + 1):] = rescaleWav(U[:2 ** j:, 2 ** j:2 ** (j + 1):])
        U[2 ** j:2 ** (j + 1):, :2 **
          j:] = rescaleWav(U[2 ** j:2 ** (j + 1):, :2 ** j:])
        U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):] = (
            rescaleWav(U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):]))
    # coarse scale
    U[:2 ** j:, :2 ** j:] = rescale(U[:2 ** j:, :2 ** j:])
    # plot underlying image
    imageplot(U)
    # display crosses
    for j in np.arange(Jmax, Jmin - 1, -1):
        plt.plot([0, 2 ** (j + 1)], [2 ** j, 2 ** j], 'r')
        plt.plot([2 ** j, 2 ** j], [0, 2 ** (j + 1)], 'r')
    # display box
    plt.plot([0, n], [0, 0], 'r')
    plt.plot([0, n], [n, n], 'r')
    plt.plot([0, 0], [0, n], 'r')
    plt.plot([n, n], [0, n], 'r')
    return U

def plot_wavelet(fW, Jmin=0):
    """
        plot_wavelet - plot wavelets coefficients.

        U = plot_wavelet(fW, Jmin):

        Copyright (c) 2020 P.-A. Thouvenin (adapted from plot_wavelet above)
        
        Bug fix: fix display inversion, coming from the output array of the p
        ywt.coeffs_to_array() function (cH and cV interverted).
    """
    def rescaleWav(A):
        v = abs(A).max()
        B = A.copy()
        if v > 0:
            B = .5 + .5 * A / v
        return B
    ##
    n = fW.shape[1]
    Jmax = int(np.log2(n)) - 1
    U = np.zeros(fW.shape)
    for j in np.arange(Jmax, Jmin - 1, -1):
    #
        # horizontal
        U[:2 ** j:,    2 ** j:2 **
            (j + 1):] = rescaleWav(fW[2 ** j:2 ** (j + 1):, :2 ** j:])
        # vertical
        U[2 ** j:2 ** (j + 1):, :2 **
          j:] = rescaleWav(fW[:2 ** j:, 2 ** j:2 ** (j + 1):])
        # diagonal
        U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):] = \
            rescaleWav(fW[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):])
    # coarse scale
    U[:2 ** j:, :2 ** j:] = rescale(fW[:2 ** j:, :2 ** j:])
    # plot underlying image
    imageplot(U)
    # display crosses
    for j in np.arange(Jmax, Jmin - 1, -1):
        plt.plot([0, 2 ** (j + 1)], [2 ** j, 2 ** j], 'r')
        plt.plot([2 ** j, 2 ** j], [0, 2 ** (j + 1)], 'r')
    # display box
    plt.plot([0, n], [0, 0], 'r')
    plt.plot([0, n], [n, n], 'r')
    plt.plot([0, 0], [0, n], 'r')
    plt.plot([n, n], [0, n], 'r')
    return U

def psnr(x, y, vmax=-1):
    """
     psnr - compute the Peack Signal to Noise Ratio

       p = psnr(x,y,vmax);

       defined by :
           p = 10*log10( vmax^2 / |x-y|^2 )
       |x-y|^2 = mean( (x(:)-y(:)).^2 )
       if vmax is ommited, then
           vmax = max(max(x(:)),max(y(:)))

       Copyright (c) 2014 Gabriel Peyre
    """

    if vmax < 0:
        m1 = abs(x).max()
        m2 = abs(y).max()
        vmax = max(m1, m2)
    d = np.mean((x - y) ** 2)
    return 10 * np.log10(vmax ** 2 / d)

def snr(x, y):
    """
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.

    Copyright (c) 2014 Gabriel Peyre
    """

    return 20 * np.log10(pylab.norm(x) / pylab.norm(x - y))
