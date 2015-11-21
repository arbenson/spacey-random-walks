import numpy as np
import sys
from simplex_projection import euclidean_proj_simplex

def Simulate(PTP):
    num_samples = 20000
    dim = PTP.shape[0]
    history = np.ones(dim)
    j = 0
    seq = []
    for _ in xrange(num_samples):
        # Choose from history
        occupancy = history / np.sum(history)
        k = np.random.choice(range(dim), p=occupancy)
        i = np.random.choice(range(dim), p=PTP[:, j, k])
        seq.append(i)
        history[i] += 1
    return seq

def NormalizeStochastic(X):
    P = np.copy(X)
    N = X.shape[0]
    for j in xrange(N):
        for k in xrange(N):
            val = X[:, j, k]
            P[:, j, k] = val / np.sum(val)
    return P

def NormalizedGradient(G):
    P = np.copy(X)
    N = X.shape[0]
    for j in xrange(N):
        for k in xrange(N):
            val = X[:, j, k]
            P[:, j, k] = val / np.sum(np.abs(val))
    return P

def EstimateSecondOrder(seq):
    dim = max(seq) + 1
    X = np.zeros((dim, dim, dim))
    for l in xrange(1, len(seq)):
        if l > 1:
            k = seq[l - 2]
        else:
            k = 0
        j = seq[l - 1]
        i = seq[l]
        X[i, j, k] += 1

    return NormalizeStochastic(X)

def Gradient(seq, X):
    ''' Compute the gradient for the current xi values. '''
    grad = np.zeros(X.shape, order='C')
    history = np.ones(X.shape[0])
    for l in xrange(0, len(seq)):
        i = seq[l]
        if l >= 1:
            j = seq[l - 1]
        else:
            j = 0
        occ_v = history / np.sum(history)
        vals = occ_v * X[i, j, :]
        grad[i, j, :] += occ_v / np.sum(vals)
        history[i] += 1
    return grad

def LogLikelihood(seq, X):
    ''' Compute the log likelihood for the current xi values. '''
    ll = 0.0
    history = np.ones(X.shape[0])
    for l in xrange(1, len(seq)):
        i = seq[l]
        j = seq[l - 1]
        occ_v = history / np.sum(history)
        trans = X[i, j, :]
        ll += np.log(np.sum(occ_v * trans))
        history[i] += 1
    return ll

def EstimateSRS(seq):
    X = EstimateSecondOrder(seq)
    dim = max(seq) + 1
    #X = NormalizeStochastic(np.ones((dim, dim, dim)))
    curr_ll = LogLikelihood(seq, X)
    print 'starting log likelihood: ', curr_ll

    def Project(Z):
        ZZ = np.copy(Z)
        N = Z.shape[0]
        for j in xrange(N):
            for k in xrange(N):
                ZZ[:, j, k] = euclidean_proj_simplex(Z[:, j, k])
        return ZZ
        

    niter = 1000
    iter = 0
    step_size = 1.0
    while iter < niter:
        grad = Gradient(seq, X)
        while step_size > 1e-12:
            X + step_size
            Y = Project(X + step_size * grad)
            next_ll = LogLikelihood(seq, Y)
            if next_ll > curr_ll:
                X = Y
                curr_ll = next_ll
                print curr_ll, step_size
                break
            else:
                step_size /= 10.0
        iter += 1
        
    return X

if __name__ == '__main__':
    # Generate the transition probabilities
    N = 4
    X = np.zeros((N, N, N))
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                X[i, j, k] = np.random.uniform(0, 1)
    P = NormalizeStochastic(X)
    seq = Simulate(P)
    PSO = EstimateSecondOrder(seq)
    PSRS = EstimateSRS(seq)

    dim = P.shape[0]
    for i in xrange(dim):
        for j in xrange(dim):
            for k in xrange(dim):
                print P[i, j, k], PSRS[i, j, k], PSO[i, j, k]

