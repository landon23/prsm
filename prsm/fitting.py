import numpy as np
import scipy.stats
import scipy.special
from .density import density
from .methods import get_func



def density_fit(xx, nbins, k, edge = None, sq = True, alpha = None):
    #this function attempts to fit a density having a power law behavior to eigenvalues supplied by xx
    # xx should be decreasing.
    # the method is to first find the empirical cdf F(x) = | { x_i : x_i < x } |, evaluated at nbins number of points
    # which are a grid evenly spaced around the minimal and maximal eigenvalues in xx.
    # then a polynomial linear regression is fit to F^(1 / (1+ alpha ))
    # k is the degree of the fitted polynomial
    #  a density.density object is returned.

    #sort in descending order.
    if (np.diff(xx) >0).any():
        xx = np.sort(xx)[::-1]


    fit_edge = (edge is None)
    if sq:
        alpha = 0.5
    if (not sq) and (alpha is None):
        print('Fit an alpha first!')
        return
    if fit_edge:
        xbulk = xx[0] - xx
    else:
        xbulk = edge - xx
    bulkmin, bulkmax = xbulk[0], xbulk[-1]

    xbulk = xbulk / xbulk[-1]

    b, a = fit_cdf_power(xbulk, nbins, fit_edge=fit_edge, k=k, alpha=alpha)

    # scaling factor calculated (see density.density object definition for details)
    if sq:
        sf = 1 / (np.power(3 / 2, 2 / 3) * a[1] / np.power((bulkmax-bulkmin), 1/3))
        sf = sf / np.power(np.pi, 2 / 3)
    else:
        sf = (alpha+1)*np.power(a[1], alpha+1) #F' = sf*(x-b)^alpha

    G, Gd = get_func(a, alpha)

    rescal = lambda x: x / bulkmax
    F = lambda x: G(rescal(x) - b) / G(1 - b)
    Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax))
    if sq:
        sf = sf* np.power(G(1-b), 2/3)
    else:
        sf = sf*np.power(bulkmax, alpha-1)
        sf = sf / G(1-b)

    if fit_edge:
        p = lambda x : Fd(xx[0]-x)
        cdf = lambda x : F(xx[0]-x)
        r = xx[0] -b * bulkmax
        l = xx[0] - bulkmax
    else:
        p = lambda x: Fd(edge - x)
        cdf = lambda x: F(edge - x)
        r = edge
        l = edge - bulkmax
    dens = density(p, l, r, F=cdf, scaling_factor = sf, sq=sq, alpha=alpha)
    return dens

def fit_cdf_power(x, n, k=1, alpha=0.5, fit_edge= True, verbose = False):
    #assume x is increasing
    # Behavior is best when x begins at 0.
    #n is number of points to evaluate the cdf on F
    #k is order of polynomial
    #alpha is power of density, p(x) ~ x^\alpha
    # if fit_edge = False, will assume that the edge is at 0, and so the polynomial fit will not include a 0th order term
    #what is returned is a scalar b, and a vector a.
    #the fit is F^(1/(1+alpha)) = \sum_{i=1}^k a_i (x-b)^i
    #note a[0] = 0.

    F, right = get_cdf(x, n)  #this returns the empirical CDF of x evaluated at n points.
    # Right is the location of the points F is evaluated at, i.e. F_i = F ( right(i)).
    # Note this function heavily relies on x increasing.

    F = F / F[-1]

    y = np.power(F, 1 / (1 + alpha))

    #try to find the root of of the fit polynomial closest to the origin.
    # if no root, return instead a linear polynomial (basically guaranteed to have a root)
    if fit_edge:
        P = np.polynomial.polynomial.Polynomial.fit(right, y, k)
        P.convert(domain=np.array([-1, 1]))
        ro = P.roots()
        if not np.isreal(ro).any():
            if verbose:
                print('no real roots found for initial fit of k= ', k,'. Instead fit k=1')
            k=1
            P = np.polynomial.polynomial.Polynomial.fit(right, y, k)
            P.convert(domain=np.array([-1, 1]))
            ro = P.roots()
        I = np.where(np.isreal(ro))[0]
        rero = ro[I]

        b = rero[np.argmin(np.abs(rero))]
        b = np.real(b)

    #fit y = a[0] (x-b) + a[1](x-b)^2 + ...
        a = np.zeros(shape=k)
        # note that the formula here is that if p(x) is a polynomial and we want to find the expansion
        # p(x) = \sum_i a_i (x-b)^i, then a_i = p^{(i)} (b) / i!
        for i in range(k):
            a[i] = (P.deriv(i+1)).__call__(b) / scipy.special.factorial(i+1)
        return b, np.append(np.array([0]), a)
    else:
        P = np.polynomial.polynomial.Polynomial.fit(right, y, np.arange(1, k+1))
        P.convert(domain=np.array([-1, 1]))
        a = P.coef
        return 0.0, a



def get_cdf(x, n):
    #x is increasing
    #n+1 is number of points to fit F
    rang = x[-1] - x[0]
    right = rang * (np.arange(n + 1) - 0.5) / n + x[0]
    F = np.zeros(shape=(n + 1))
    j = 0
    current = x[0]
    for i in range(n + 1):
        edge = right[i]
        if i > 0:
            F[i] = F[i - 1]

        while current < edge:
            F[i] += 1
            j += 1
            if j < len(x):
                current = x[j]
            else:
                current = edge[n] + 10
    return F, right



def power_grid(x, n, betas):
    #betas is a grid of exponenets to check.
    # Finds empirical cdf of x, F
    # Exponentiates   y = F^(1/beta)
    # linear regression to y, report residiual (normalized)
    # returns residuals.

    F, right = get_cdf(x, n)
    I = np.where(F>0)[0]
    F = F[I]
    F = F / F[-1]
    right = right[I]
    resids = np.zeros(len(betas))

    for i in range(len(betas)):
        f = np.power(F, 1 / betas[i])
        result = scipy.stats.linregress(right, f)
        sl = result[0]
        inte = result[1]
        pr = inte + sl*right
        resids[i] = np.mean(np.power(f-pr, 2.0)) /np.var(f)
    return resids


#This fitting method is likely obsolete.
def square_root_fit(xx, nbins, k, edge = None):
    fit_edge = (edge is None)
    if fit_edge:
        xbulk = xx[0]-xx
    else:
        xbulk = edge - xx

    bulkmin, bulkmax = xbulk[0], xbulk[-1]

    xbulk = xbulk / xbulk[-1]

    b, a = fit_cdf_power(xbulk, nbins,  fit_edge=fit_edge, k=k)
    sf = 1 / ( np.power(3/2, 2/3)*a[1]/ (bulkmax-bulkmin))
    sf = sf / np.power(np.pi, 2/3)

    G, Gd = get_func(a, 0.5)

    #rescal = lambda x: (x - bulkmin) / (bulkmax - bulkmin)
    rescal = lambda x: x / bulkmax
    F = lambda x: G(rescal(x) - b) / G(1 - b)
    #Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax - bulkmin))

    Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax))
    if fit_edge:
        p = lambda x : Fd(xx[0]-x)
        cdf = lambda x : F(xx[0]-x)
        r = xx[0] -b * (bulkmax - bulkmin) + bulkmin
        l = xx[0] - bulkmax
    else:
        p = lambda x: Fd(edge - x)
        cdf = lambda x: F(edge - x)
        r = edge
        l = edge - bulkmax

    dens = density(p, l, r, F=cdf, scaling_factor = sf)
    return dens