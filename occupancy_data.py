import numpy as np
import cPickle as pickle
import sys

def GetSeqs(filename):
    seqs = []
    with open(filename) as f:
        for line in f:
            seq = [int(x) for x in line.strip().split(',')]
            seqs.append(seq)
    return seqs

def MaxInd(seqs):
    max_ind = 0
    for seq in seqs:
        for ind in seq:
            max_ind = max(max_ind, ind)
    return max_ind

def OccVectors(seqs):
    dim = MaxInd(seqs) + 1
    P = np.ones((len(seqs), dim))
    for i, seq in enumerate(seqs):
        for ind in seq:
            P[i][ind] += 1

    for i in xrange(len(seqs)):
        P[i, :] = P[i, :] / np.sum(P[i, :])
    return P

if __name__ == '__main__':
    seqs = GetSeqs(sys.argv[1])
    P = OccVectors(seqs)
    with open(sys.argv[2], 'w') as f:
        pickle.dump(P, f)
    print P
