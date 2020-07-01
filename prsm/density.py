import numpy as np
import scipy.integrate
import scipy.special


class density:
    #density is meant to hold a probability density fit near the edge of the bulk of the eigenvalues

    def __init__(self, p, l, r, F=None, scaling_factor = None, sq=True, alpha = None):
        self.p =p
        # p is the probability density function p(x).  Intended to integrate to 1 (even though it is not fit to all the eigenvalues)
        # its support is the interval (l, r)
        self.l = l
        self.r = r
        self.F = F #optional CDF
        self.scaling_factor = scaling_factor
        # scaling factor is slightly different in square root case.  If sf = True,
        # then p(x) ~ (pi)^{-1} (r-x)^{1/2} dx / (sf)^{3/2}.   Then for the TW distribution, lambda_1 ~ r + sf * N^{-2/3} TW
        # if sf = False then p(x) ~ sf*(r-x)^alpha
        self.sq = sq
        if not sq:
            self.alpha = alpha
        else:
            self.alpha = 0.5



class approximate_esd:

    #this holds a full approximate spectral distribution, consisting of two components
    # 1) a fitted edge density
    # 2) the remaining eigenvalues further from the edge.

    def __init__(self, dens, bulkev, n):
        #dens is a density as above, bulkev all of the non-outlier eigenvalues.
        # Note that the convention is that all of the eigenvalues are passed in bulkev, even if some will not be used
        # n is the eigenvalues that the density is meant to represent
        self.dens = dens
        self.bulkev = bulkev
        self.n = n
        self.wr = n / (len(bulkev.flatten()))
        self.wl = 1 - self.wr
        if dens.sq:
            if self.dens.scaling_factor is not None:
                self.sf = self.dens.scaling_factor / np.power(self.wr, 2/3)
        else:
            M = len(bulkev.flatten())
            alpha = self.dens.alpha
            sf = self.wr*self.dens.scaling_factor

            self.ip = np.power((1+alpha)/(sf*M), 1 / (1+alpha))
    def calc_m(self, E, k=0):
        f = lambda x : 1 / (x-E)
        F = lambda x : np.power(f(x), k+1)*scipy.special.factorial(k)
        p = self.dens.p
        g = lambda x : p(x) * F(x)
        cr = self.wr*scipy.integrate.quad(g, self.dens.l, self.dens.r)[0]
        cl = self.wl*np.mean(F(self.bulkev[self.n:]))
        return cl+cr