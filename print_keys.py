import cPickle as pickle
import sys

with open(sys.argv[1]) as f:
    data = pickle.load(f)
    kv = sorted(data.iteritems(), key=lambda x: x[1])
    kv.sort()
    for k, v in kv:
        print k, v

    #vals = [k for k, v in kv]
    #print ','.join(vals)

    
