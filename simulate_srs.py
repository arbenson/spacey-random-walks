import numpy as np
import sys
from simplex_projection import euclidean_proj_simplex

def Simulate(PTP, num_samples):
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
        j = i
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

def EstimateSecondOrder(seqs):
    dim = max([max(seq) for seq in seqs]) + 1
    X = np.zeros((dim, dim, dim))
    for seq in seqs:
        for l in xrange(1, len(seq)):
            if l > 1:
                k = seq[l - 2]
            else:
                k = 0
            j = seq[l - 1]
            i = seq[l]
            X[i, j, k] += 1

    return NormalizeStochastic(X)

def Gradient(seqs, X):
    ''' Compute the gradient for the current xi values. '''
    grad = np.zeros(X.shape, order='C')
    for seq in seqs:
        history = np.ones(X.shape[0])
        for l in xrange(0, len(seq)):
            i = seq[l]
            if l >= 1:
                j = seq[l - 1]
            else:
                j = 0
            occupancy = history / np.sum(history)
            vals = occupancy * X[i, j, :]
            grad[i, j, :] += occupancy / np.sum(vals)
            history[i] += 1

    return grad

def LogLikelihood(seqs, X):
    ''' Compute the log likelihood for the current xi values. '''
    ll = 0.0
    for seq in seqs:
        history = np.ones(X.shape[0])
        for l in xrange(1, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            occupancy = history / np.sum(history)
            trans = X[i, j, :]
            ll += np.log(np.sum(occupancy * trans))
            history[i] += 1
    return ll

def EstimateSRS(seqs):
    #X = EstimateSecondOrder(seqs)
    #dim = X.shape[0]
    dim = max([max(seq) for seq in seqs]) + 1
    X = NormalizeStochastic(np.ones((dim, dim, dim)))
    curr_ll = LogLikelihood(seqs, X)
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
    #step_size = 1e-3
    while iter < niter:
        step_size = 1e-3 / (iter + 1)
        grad = Gradient(seqs, X)
        Y = Project(X + step_size * grad)
        next_ll = LogLikelihood(seqs, Y)
        if next_ll > curr_ll:
            X = Y
            curr_ll = next_ll
            print curr_ll, step_size
        iter += 1
        
    return X

def SecondOrderLogLikelihood(seqs, P):
    ll = 0.0
    for seq in seqs:
        for l in xrange(1, len(seq)):
            if l > 1:
                k = seq[l - 2]
            else:
                k = 0
            j = seq[l - 1]
            i = seq[l]
            ll += np.log(P[i, j, k])
    return ll

if __name__ == '__main__':
    # Generate the transition probabilities
    N = 4
    num_seqs = 20
    samples_per_seq = 500

    X = np.zeros((N, N, N))
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                X[i, j, k] = np.random.uniform(0, 1)

    P = NormalizeStochastic(X)
    seqs = [Simulate(P, samples_per_seq) for _ in xrange(num_seqs)]
    PSO = EstimateSecondOrder(seqs)
    print 'Second-order log likelihood', SecondOrderLogLikelihood(seqs, P)
    print 'Oracle log likelihood', LogLikelihood(seqs, P)
    PSRS = EstimateSRS(seqs)
    # Likelihood of second-order model

    dim = P.shape[0]
    for i in xrange(dim):
        for j in xrange(dim):
            for k in xrange(dim):
                print P[i, j, k], PSRS[i, j, k], PSO[i, j, k]

