import sys
import numpy as np
import cPickle as pickle

def FirstOrderLogLikelihood(seqs, P1):
    ''' Compute the log likelihood for the current xi values. '''
    def SeqLogLikelihood(seq):
        seq_ll = 0.0
        for l in xrange(2, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            seq_ll += np.log(P1[i, j])
        return seq_ll

    ll = 0.0
    for seq in seqs:
        ll += SeqLogLikelihood(seq)
    return ll

def SecondOrderLogLikelihood(seqs, P2):
    ''' Compute the log likelihood for the current xi values. '''
    def SeqLogLikelihood(seq):
        seq_ll = 0.0
        for l in xrange(2, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            k = seq[l - 2]
            seq_ll += np.log(P2[i, j, k])
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

def FirstOrderMarkovP(seqs, N):
    P = np.zeros((N, N))
    for seq in seqs:
        for l in xrange(1, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            P[i, j] += 1
    
    for j in xrange(N):
        if np.sum(P[:, j]) == 0.0:
            P[:, j] = np.ones(N)
        P[:, j] = P[:, j] / np.sum(P[:, j])
    return P

def SecondOrderMarkovP(seqs, N):
    P = np.zeros((N, N, N))
    for seq in seqs:
        for l in xrange(2, len(seq)):
            i = seq[l]
            j = seq[l - 1]
            k = seq[l - 2]
            P[i, j, k] += 1
    
    for j in xrange(N):
        for k in xrange(N):
            if np.sum(P[:, j, k]) == 0.0:
                P[:, j, k] = np.ones(N)
            P[:, j, k] = P[:, j, k] / np.sum(P[:, j, k])
    return P


if __name__ == '__main__':
    seqs = LoadSeqs(sys.argv[1])
    print len(seqs)
    max_ind = MaxIndex(seqs) + 1
    P1 = FirstOrderMarkovP(seqs, max_ind)
    ll1 = FirstOrderLogLikelihood(seqs, P1)
    print ll1
    P2 = SecondOrderMarkovP(seqs, max_ind)
    ll2 = SecondOrderLogLikelihood(seqs, P2)
    print ll2
    
