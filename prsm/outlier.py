
from prsm.methods import calc_outlier_quantities


class outlier:
    #class to hold the value and properties of an outlying eigenvalue. Also can calculate the properties of the outlier.
    def __init__(self, N, M, sample = None, population = None, samp_std = None, overlap = None, over_norm_std = None, index = None):
        self.N = N
        self.M = M
        self.gam = M/ N
        self.sample = sample
        self.population = population
        self.samp_std = samp_std
        self.overlap = overlap
        self.over_norm_std = over_norm_std
        self.index = index
        self.inside_spec = False
    def report(self, give_over_std = True):
        #format is:
        #(Boolean whether eigenvalue is inside spectrum, sample eigenvalue value, estimated population eigenvalue, estimated sample std, estimated overlap, esimated overlap standard error)

        return (self.inside_spec, self.sample, self.population, self.samp_std, self.overlap, self.over_norm_std)


    def report_edge_dist(self, b, L, tw=True):
        if tw:
            print('Distance from TW-mean normalized by TW-std:', b/L)
        else:
            print('Distance from edge normalized by density inter-particle distance', b/L)

    def calculate(self, data):
        self.population, self.samp_std, self.overlap, self.over_norm_std = calc_outlier_quantities(self.N, self.M, data)