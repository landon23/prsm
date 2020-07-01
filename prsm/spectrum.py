
import numpy as np
import matplotlib.pyplot as plt

from prsm.fitting import density_fit, power_grid
from prsm.density import approximate_esd
from prsm.outlier import outlier
from prsm.methods import print_outlier_quants, test_outlier



class spectrum:

    def __init__(self, S, N, M, nout = None, fit_power_law = False):
        #Expects S to be the M eigenvalues of N^{-1} X^T X, where X is an N x M data matrix (N independent samples of M-dimensional data)
        #S should be in decreasing order.
        #nout can be passed to hard-set the number of outliers.  They will be skipped when using the fit(), find_alpha() methods
        if len(S) != N and len(S) != M:
            print('S is wrong length')
            return
        if not (-np.diff(S) <= 0).all():
            S = np.sort(S)[::-1]  #sort S if not already sorted in descending order.
        if len(S) == N and N > M:
            S = S[0:M]
        elif len(S) == N and N<M:
            S = np.append(S, np.zeros(M-N))

        self.eigenvalues = S
        self.N = N
        self.M = M
        self.gam = M/N
        self.nout = nout
        if not fit_power_law:
            self.alpha =0.5
        else:
            self.alpha = None
        self.sq = not fit_power_law
    def fit(self,  nbins, n, k, nout=None, edge = None):
        #nout = number of outliers
        #n = number of ev to fit density
        #k degree of polynomial to use in fit.
        # if edge is passed, then it will be used, and an edge will not be fit
        self.nfit = n
        if nout is None:
            nout  = self.nout

        if nout is None:
            print('Supply number of outliers')
        else:
            if self.sq:
                self.edge_density = density_fit(self.eigenvalues[nout:nout+n], nbins, k, alpha = 0.5, sq=True, edge= edge)
                self.appr_esd = approximate_esd(self.edge_density, self.eigenvalues[nout:], n)
                self.tw_mean = self.appr_esd.dens.r -1.2065336*self.appr_esd.sf / np.power(self.M, 2/3)
                self.tw_std = np.sqrt(1.60778)*self.appr_esd.sf / np.power(self.M, 2/3)
            else:
                if self.alpha is None:
                    print('Please fit or set self.alpha')
                    return
                self.edge_density = density_fit(self.eigenvalues[nout:nout+n], nbins, k, alpha=self.alpha, sq=False, edge=edge)
                self.appr_esd=approximate_esd(self.edge_density, self.eigenvalues[nout:], n)
                self.edge = self.edge_density.r
                self.ip = self.appr_esd.ip


    def calc_outlier_quants(self, nout=None):
        # calculates various outlier quantities - estimated population eigenvalues, overlap, and standard deviations, etc.
        # stored in the list self.outliers whose entries are the outlier class.
        if nout is None:
            nout = self.nout
        if nout is None:
            print('Supply number of outliers')
        else:
            self.outliers = []
            for i in range(nout):
                s = self.eigenvalues[i]

                self.outliers.append( outlier(self.N, self.M, sample = s))
                if s < self.appr_esd.dens.r:
                    self.outliers[i].inside_spec = True
                else:
                    self.outliers[i].inside_spec = False
                    m = self.appr_esd.calc_m(s, 0)
                    mp = self.appr_esd.calc_m(s, 1)
                    mppp = self.appr_esd.calc_m(s, 3)
                    data = (s, m, mp, mppp)
                    self.outliers[i].calculate(data)

    def report(self, verbose = True):
        #returns various outlier quantities to a list of tuples, in format:
        #(Boolean whether eigenvalue is inside spectrum, sample eigenvalue value, estimated population eigenvalue, estimated sample std, estimated overlap, esimated overlap standard error)

        x = []
        for i in range(self.nout):
            x.append(self.outliers[i].report())
            if verbose:
                print_outlier_quants(x[i])
        return x


    def outlier_diagnostics(self, verbose = True):
        # for each outlier, calculates some diagnostics which allow to assess whether the eigenvalue is an outlier
        # returns a list of tuples, one tuple per eigenvalue
        #format of tuples is:
        # (eigenvalue location, distance to nearest neighboring eigenvalue divided by sample eigenvale standard deviation,
        # distance to spectral edge divided by sample eigenvalue std, distance to edge divided by interparticle distance)
        #in square root case, final quantity is distance to the mean of a TW distribution fitted to spectral edge, divided by TW std
        x = []
        for i in range(self.nout):
            outlier = self.outliers[i]
            if i == 0:
                d = self.eigenvalues[0]-self.eigenvalues[1]
            else:
                d = np.minimum(self.eigenvalues[i-1] - self.eigenvalues[i], self.eigenvalues[i]-self.eigenvalues[i+1])
            s = outlier.sample
            nearest  = d / outlier.samp_std
            edge_dist = (outlier.sample-self.appr_esd.dens.r) / outlier.samp_std

            if self.sq:
                q = (outlier.sample - self.tw_mean)/self.tw_std
            else:
                q = (outlier.sample - self.appr_esd.dens.r)/ self.appr_esd.ip
            tup = (s, nearest, edge_dist, q)
            x.append(tup)
            if verbose:
                print('Outlier with index', i, ' at location: ', s)
                print('Distance to nearest neighbouring eigenvalue normalized by outlier std: ', nearest)
                print('Distance to spectral edge normalized by outlier std: ', edge_dist)
                if self.sq:
                    print('Distance to TW mean normalized by TW std:', q)
                else:
                    print('Distance to spectral edge normalized by density interparticle distance', q)

        return x

    def fit_alpha(self, n, m, alphas):
        #accepts an array of alphas and attempts to find best fitting power law using residuals.  Also returns residuals
        #more details on fitting in definition of methods.power_grid
        if self.nout is None:
            print('please set number of outliers')
            return
        x = self.eigenvalues[self.nout:self.nout+n]
        x = x[0]-x
        resid = power_grid(x, m, 1+alphas)
        i = np.argmin(resid)
        self.alpha = alphas[i]
        return resid





    def auto_sq_fit(self, nbins, n, k, n_init=0, edge_thresh = 4.0, over_thresh = 0.1, nmax = None, supplied_density = None):
        #iterative algorithm for finding the number of outliers when fitting to a square-root
        #on run n_out, first fits square root density to the n eigenvalues starting with the n_out+1st largest one
        #then checks if any the first n_out eigenvalues violate any of the outlier thresholds, given by edge_thresh (how far from the edge) and over_thresh (size of overlap standard error)

        # if a density is supplied, doesn't fit a density at each stage, just continues until an eigenvalue fails to be an outlier

        #n_init may be passed to assume a few eigenvalues are outliers - they will be tested anyway so it is possible that it stops after one iteration
        if nmax is None:
            nmax = int(self.M/4)
        cont = True
        self.nfit = n
        old_dens = density_fit(self.eigenvalues[0:n], nbins, k, sq=True)
        old_esd = approximate_esd(old_dens, self.eigenvalues[0:], n)


        nout = n_init+1
        calc_density = (supplied_density is None)


        while cont:
            if calc_density:
                edge_density = density_fit(self.eigenvalues[nout:n+nout], nbins, k, sq=True)
                appr_esd = approximate_esd(edge_density, self.eigenvalues[nout:], n)
            else:
                edge_density = supplied_density.dens
                appr_esd = supplied_density
            tw_mean = appr_esd.dens.r - 1.2065336 * appr_esd.sf / np.power(self.M, 2 / 3)
            tw_std = np.sqrt(1.60778) * appr_esd.sf / np.power(self.M, 2 / 3)
            if calc_density:
                init = 0
            else:
                init = nout-1
            for i in range(init, nout):
                s = self.eigenvalues[i]
                if cont:
                    if s <= edge_density.r:
                        cont = False
                        ifail = i
                        message = 'Eigenvalue within spectral distribution'

                    elif nout >= nmax:
                        cont = False
                        ifail = -1
                        message = 'Max outliers reached'


                    else:
                        possible_outlier = outlier(self.N, self.M, sample = s)
                        is_outlier, message = test_outlier(possible_outlier, appr_esd, edge_thresh, over_thresh, tw_mean, tw_std)
                        if is_outlier == False:
                            cont = False
                            ifail = i

            if cont:
                nout = nout+1
                old_dens = edge_density
                old_esd = appr_esd
            else:
                nout = nout-1
                print('Number of outliers found: ', nout)
                print('Index of eigenvalue that failed test: ', ifail)
                print('Reason: '+message)
                self.nout = nout
                self.edge_density = old_dens
                self.appr_esd = old_esd
                self.tw_mean = old_esd.dens.r - 1.2065336 * old_esd.sf / np.power(self.M, 2 / 3)
                self.tw_std = np.sqrt(1.60778) * old_esd.sf / np.power(self.M, 2 / 3)

    #plotting tools
    def plot_density(self, grid_size=1000, nbins=30):
        #plots the density against the eigenvalues it fit
        x = (np.arange(grid_size)/ grid_size)*(self.edge_density.r - self.edge_density.l)+self.edge_density.l
        y = self.edge_density.p(x)
        x = np.append(x, np.array([self.edge_density.r]))
        y = np.append(y, np.array([0.0]))
        plt.plot(x, y, color='red')
        plt.hist(self.eigenvalues[self.nout:self.nfit+self.nout], bins = nbins, density = True, alpha = 0.5)
        plt.show()

    def plot_sm(self, grid_size = 1000, nbins = 30):
        x = (np.arange(grid_size) / grid_size) * (self.edge_density.r - self.edge_density.l) + self.edge_density.l
        y = self.edge_density.p(x)
        x = np.append(x, np.array([self.edge_density.r]))
        y = np.append(y, np.array([0.0]))*self.appr_esd.wr

        cut = self.eigenvalues[self.nout+self.nfit]
        fig, ax = plt.subplots()
        nn, bins, patches = ax.hist(self.eigenvalues[self.nout:], bins= nbins, density = True, alpha=0.8)
        ii = np.argmax(bins > cut)
        for i in range(ii, nbins):
            patches[i].set_color('#38ACEC')
        for i in range(0, ii):
            patches[i].set_color('#157DEC')

        plt.plot(x, y, color='red')
        plt.show()

