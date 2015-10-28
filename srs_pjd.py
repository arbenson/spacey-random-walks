import sys
import numpy as np
import cPickle as pickle

def Normalize(Z):
    max_ind = Z.shape[0]
    for j in xrange(max_ind):
        for k in xrange(max_ind):
            vals = np.exp(Z[:, j, k])
            Z[:, j, k] = np.log(vals / np.sum(vals))

def Gradient(seqs, X):
    ''' Compute the gradient for the current xi values. '''
    def SeqGrad(seq):
        seq_grad = np.zeros(X.shape, order='C')
        occ_s = np.ones(X.shape[0])
        for l in xrange(1, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            occ_v = occ_s / np.sum(occ_s)
            vals = occ_v * np.exp(X[i, j, :])        
            seq_grad[i, j, :] += vals / np.sum(vals)
            occ_s[i] += 1
        return seq_grad

    grad = np.zeros(X.shape, order='C')
    for seq in seqs:
        grad += SeqGrad(seq)
    return grad

def LogLikelihood(seqs, X):
    ''' Compute the log likelihood for the current xi values. '''
    def SeqLogLikelihood(seq):
        seq_ll = 0.0
        occ_s = np.ones(X.shape[0])
        for l in xrange(1, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            occ_v = occ_s / np.sum(occ_s)
            trans = np.exp(X[i, j, :])
            seq_ll += np.log(np.sum(occ_v * trans))
            occ_s[i] += 1
        return seq_ll

    ll = 0.0
    for seq in seqs:
        ll += SeqLogLikelihood(seq)
    return ll

def LoadSeqs(filename):
    seqs = []
    with open(filename) as f:
        for line in f:
            seq = [int(x) for x in line.split(',')]
            seq = np.array(seq)
            seqs.append(seq)
    return seqs

def MaxIndex(seqs):
    max_val = 0
    for seq in seqs:
        max_val = max(max_val, np.max(seq))
    return max_val    

def LearnProbs(seqs, X):
    curr_ll = LogLikelihood(seqs, X)
    print 'starting log likelihood: ', curr_ll
    Normalize(X)

    niter = 200
    iter = 0
    while iter < niter:
        gamma = 1.0
        grad = Gradient(seqs, X)
        while gamma > 1e-10:
            Y = np.minimum(X + gamma * grad, -1e-6) 
            Normalize(Y)
            next_ll = LogLikelihood(seqs, Y)
            if next_ll > curr_ll:
                X = Y
                curr_ll = next_ll
                print curr_ll, gamma
                break
            else:
                gamma /= 10.0
        iter += 1

    # Normalize probability tensor
    max_ind = X.shape[0]
    P = np.zeros(X.shape)
    for j in xrange(max_ind):
        for k in xrange(max_ind):
            curr_vals = np.exp(X[:, j, k])
            P[:, j, k] = curr_vals / np.sum(curr_vals)
    return P

if __name__ == '__main__':
    seqs = LoadSeqs(sys.argv[1])
    print len(seqs)
    max_ind = MaxIndex(seqs) + 1
    X = -np.ones((max_ind, max_ind, max_ind), order='C')

    for j in xrange(max_ind):
        for k in xrange(max_ind):
            vals = np.exp(X[:, j, k])
            X[:, j, k] = np.log(vals / np.sum(vals))
    print 'normalized...'
    P = LearnProbs(seqs, X)
    with open('learned_P.pkl', 'w') as f:
        pickle.dump(P, f)

    for k in xrange(max_ind):
        print P[:, :, k]
