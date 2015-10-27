import numpy as np
import sys

p = float(sys.argv[1])
out1 = sys.argv[2]
out2 = sys.argv[3]

with open(out1, 'w') as train:
    with open(out2, 'w') as test:
        for line in sys.stdin:
            if np.random.rand() < p:
                train.write(line)
            else:
                test.write(line)
    
