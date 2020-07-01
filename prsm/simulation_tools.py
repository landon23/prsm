import numpy as np

def make_data(N, M, sigmas):
    Z = np.random.normal(size = (N, M)) / np.sqrt(N)
    Z = Z * np.sqrt(sigmas)
    return Z

def diagonalize_cov(X):
    U, S, V = np.linalg.svd(X)  #X is N x M. U is N x N, V is M x M.
    S = S*S
    return U, S, V

def samples(N, M, sigmas, n=1):
    u = np.zeros(shape = (N, N, n))
    v = np.zeros(shape = (M, M, n))
    s = np.zeros(shape=(M, n))

    for i in range(n):
        U, S, V = diagonalize_cov(make_data(N, M, sigmas))
        u[:,:,i]=U
        if len(S) < M:
            S = np.append(S, np.zeros(shape=M-len(S)))
        s[:,i]=S
        v[:,:,i]=V
    if n ==1:
        return u[:,:,0], s[:,0], v[:,:,0]
    else:
        return u, s, v


def m(s, gamma):
    # for 0 < gamma < 1,
    # m(s) = (2s)^{-1} ( gamma^{-1} -1 - s / gamma + gamma^{-1} [ (s- lambda_+ ) ( s - lambda_-) ]^{1/2} )
    #with lambda_\pm = (1 \pm (gamma)^{1/2} )^2
    lamr = (1+np.sqrt(gamma))**2
    laml = (1 - np.sqrt(gamma))**2
    return (1/ (2*s))*((1/gamma)-1-(s / gamma)+ (1/gamma)*np.sqrt((s-lamr)*(s-laml)))

def mtil(s, gamma):
    return m(s, gamma)*gamma + (gamma-1) / s


