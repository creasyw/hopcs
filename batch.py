import numpy as np
from maest import maestx
import sys

result3 = []
result4 = []
step = 512
pcs = [2,3,5,7]

for k in range(0,10):
    signal = np.load("data/exp_deviate_one_%d.npz.npy"%(k))
    y = np.zeros(len(signal)-2)
    b1 = -2.333
    b2 = 0.667
    for i in range(len(y)):
        y[i] = signal[i+2]+b1*signal[i+1]+b2*signal[i]
    
    result3.append(maestx(y, pcs, 2, 3, step))
    result4.append(maestx(y, pcs, 2, 4, step))
    sys.stdout.write("%sth data file is complete, 3rd-order=%s, 4th-order=%s\r"%(k,result3[-1],result4[-1]))
    sys.stdout.flush()

f = open("test_result.txt", 'w')
for i in result3:
    print>>f, i
print>>f, "\n\n"
for i in result4:
    print>>f, i
f.close()

