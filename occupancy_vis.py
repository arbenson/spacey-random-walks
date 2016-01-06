import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cPickle as pickle
import sys
from matplotlib.colors import LogNorm

def OccVectors(seqs):
    dim = MaxInd(seqs) + 1
    P = np.ones((len(seqs), dim))
    for i, seq in enumerate(seqs):
        for ind in seq:
            P[i][ind] += 1

    for i in xrange(seqs):
        P[i, :] = P[i, :] / np.sum(P[i, :])
    return P

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        P = pickle.load(f)
    print P
    plt.imshow(P[0:100].T, cmap=cm.Greys, norm=LogNorm(vmin=1e-8, vmax=1))
    plt.colorbar()
    plt.show()
