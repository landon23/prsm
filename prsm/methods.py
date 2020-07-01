import numpy as np
import scipy.integrate

def test_outlier(out, asd, edge_thresh, over_thresh, tw_mean, tw_std):
    #checks whether an outlier is an actual outlier or not. out is of class .outlier.outlier
    #asd is of class .density.approximate_esd
    # function returns a Boolean whether an outlier or not, and returns a message of which test it failed.
    s = out.sample
    m = asd.calc_m(s, 0)
    mp = asd.calc_m(s, 1)
    mppp = asd.calc_m(s, 3)
    data = (s, m, mp, mppp)
    out.calculate(data)

    if s <= asd.dens.r:
        is_outlier = False
        message = 'Eigenvalue inside fitted spectral distribution'
    else:
        d = s - asd.dens.r
        if (d / out.samp_std) < edge_thresh:
            is_outlier = False
            message = 'Eigenvalue within threshold sample stds of spectral edge'
        elif (s - tw_mean) / tw_std < edge_thresh:
            is_outlier = False
            message = 'Eigenvalue within threshold TW stds of spectral edge'
        elif out.over_norm_std > over_thresh:
            is_outlier = False
            message = 'Normalized overlap std above threshold'
        else:
            is_outlier = True
            message = 'Passed tests'
    return is_outlier, message


def calc_outlier_quantities(N, M, data):
    #calculates the various quantities associated with outliers using the sample eigenvalue location and
    # the Stieltjes transform (usually calculated from a .density.approximate_esd object)
    #returned quantities are estimated population eigenvalue, estimated std of sample eigenvalue, estimated overlap,
    # estimated standard error of overlap
    (s, m, mp, mppp) = data
    gam = M / N
    mtil = gam*m +(gam-1) / s
    population = - 1 / mtil

    if mp is not None:
        mptil = gam*mp + (1-gam) / np.power(s, 2)
        samp_std = np.sqrt(2 / (N*mptil))
        overlap = -1.0 * mtil / (s *mptil)
    else:
        samp_std = None
        overlap = None

    if mppp is not None:
        mppptil = mppp*gam + 6*(1-gam) / np.power(s, 4.0)
        over_norm_std = s * np.sqrt(mppptil / (3*N))
    else:
        over_norm_std = None

    return population, samp_std, overlap, over_norm_std


def get_func(a, alpha):
    #let p be the polynomial p(x) = \sum_{n>0} a_{n} x^n
    #returns function objects for x -> F(x) = (\max p(x), 0 )^{alpha+1} and
    # F'(x) = (1 + alpha) p'(x) (p(x) )^alpha (again up to taking maxes)
    p = np.polynomial.polynomial.Polynomial(a)
    F = lambda x : np.power(np.maximum(0.0, p(x)), 1+alpha)
    pd = p.deriv(1)
    Fd = lambda x : (1+alpha)*np.power(np.maximum(0.0, p(x)), alpha)*np.maximum(0.0, pd(x))
    return F, Fd



def integrator(eigs, n, p, l, r, f):
    #eigs are the bulk eigenvalues (no outliers), n is the number of eigenvalues represented by p(x) a probability density
    #p a density (integrates to 1) supported in (l, r).
    #f the function to integrate.
    # returns \int f(x) d nu (x)  where nu(x) = wl * \sum_{i > n } \delta_{ \lambda_i } (x) \d x + wr * p(x) \d x
    # and wr+wl =1, wr = n / length(eigs)

    lefteigs = eigs[n:]
    wl = len(lefteigs) / len(eigs)
    wr = 1 - wl

    F = lambda x: wr*f(x)*p(x)
    cr = scipy.integrate.quad(F, l, r)[0]
    cl = wl*np.mean(f(lefteigs))
    return cr+cl



def print_outlier_quants(tup):
    if tup[0]:
        print('Eigenvalue at: ', tup[1], 'inside spectrum')
    else:
        print('Location: ', tup[1])
        print('Population eigenvalue estimate: ', tup[2])
        print('Sample eigenvalue std:', tup[3])
        print('Estimated overlap:', tup[4])
        print('Overlap standard error:', tup[5])

#below are functions which will return function objects evaluating
# 1) the distance from the spectral edge normalized by the estimated sample outlier std at that point
# 2) the overlap standard error as a function of sample eigenvalue position
# note the first two returned function objects can only take scalars due to the reliance on scipy.integrate which only
# integrates scalar value functions
# wrapper takes a function object p that takes scalar arguments and returns a function capable of taking np.arrays,
# evaluated elementwise.

def make_outlier_dist_sv(appr_esd, N, M):
    #returns the function s --> (s-R) / outlier std
    R = appr_esd.dens.r
    gam = M / N
    mp = lambda x : appr_esd.calc_m(x, k=1)
    mptil = lambda x : gam*mp(x) + (1-gam) / np.power(x, 2.0)
    std = lambda x : np.sqrt( 2 / (N*mptil(x)))

    return lambda x : (x-R) / std(x)

def make_overlap_sterror_sv(appr_esd, N, M):
    gam = M/N
    mppp = lambda x : appr_esd.calc_m(x, k=3)
    mppptil = lambda x : gam*mppp(x) + (1-gam)*6 / np.power(x, 4.0)
    return lambda x : x * np.sqrt(mppptil(x) / (3*N))

def wrapper(p):

    def f(x):
        sh = x.shape

        x = x.flatten()
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = p(x[i])
        y = y.reshape(sh)
        return y
    return f


def bin_edges(x, n, nb):
    #function for returning the edges of nb bins which have an edge at x[n]
    #x in decreasing order.
    r = x[0]
    l = x[-1]
    mid = x[n]
    fr = (r-mid)/(r-l)
    nr = int(fr*nb)+1
    nr = min(nr, nb-1)
    nl = nb - nr
    wid = max((r-mid)/nr, (mid-l) / (nl))
    edges = np.arange(nl+1)*wid+ mid-nl*wid
    edges2 = (1+np.arange(nr))*wid+mid
    return np.concatenate((edges, edges2)), nl



