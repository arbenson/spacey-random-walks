from collections import Counter, defaultdict
import sys

counts = defaultdict(Counter)

max_ind = 0
for line in sys.stdin:
    seq = [int(x) for x in line.split(',')]
    for l in xrange(3, len(seq)):
        k, j, i = seq[l-3:l]
        max_ind = max(max_ind, i, j, k)
        if min(i, j, k) > 0:
            counts[(k, j)][i] += 1

#num_items = max_ind + 1
num_items = max_ind
sys.stderr.write('Number of items: %d\n' % num_items)
num_points = sum([len(v) for v in counts.itervalues()])
sys.stderr.write('Number of nonzeros: %d\n' % num_points)
for key, cntr in counts.iteritems():
    k, j = key
    for i, cnt in cntr.iteritems():
        # Transition k -> j -> i
        #print i + 1, j + 1 + k * num_items, cnt
        print i, j + (k - 1) * num_items, cnt
