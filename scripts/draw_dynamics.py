import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='muted')

# Draw the dynamics of the 2x2x2x2 example in the paper

def Perron(R, x):
    xvec = np.array([[x], [1 - x]])
    R1 = np.dot(R, np.kron(np.kron(xvec, xvec), np.identity(len(xvec))))
    p = R1[0][0]
    q = R1[1][1]
    return (1 - q) / (2 - p - q) - x

R = np.array([[0.925, 0.925, 0.925, 0.075, 0.925, 0.075, 0.075, 0.075],
              [0.075, 0.075, 0.075, 0.925, 0.075, 0.925, 0.925, 0.925]])
xvals = list(np.arange(0, 1, .001))
yvals = [Perron(R, x) for x in xvals]
ind1 = xvals.index(0.097)
ind2 = xvals.index(0.5)
ind3 = xvals.index(0.897)

fsz = 24
ms = 20
plt.plot(xvals, yvals, lw=4, color='#7570b3')
plt.plot(xvals[ind1], yvals[ind1], 'o', ms=ms, color='#1b9e77')
plt.plot(xvals[ind3], yvals[ind3], 'o', ms=ms, color='#1b9e77')
plt.plot(xvals[ind2], yvals[ind2], 's', ms=ms, color='#d95f02')

plt.xlabel('x', fontsize=fsz+6)
plt.ylabel('f(x)', fontsize=fsz+6)
plt.tick_params(labelsize=fsz)

sns.despine()
plt.savefig('srs_dynamics.pdf', bbox_inches='tight')
plt.show()
