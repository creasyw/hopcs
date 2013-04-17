import numpy as np
from maest import maest
import sys

y = np.load("data/exp_deviate_one_0.npz.npy")

i = 128
result3 = []
result4 = []
while i < len(y):
    result3.append(maest(y, 2, 3, i))
    result4.append(maest(y, 2, 4, i))
    i *= 2
    sys.stdout.write("Now the %s length is complete\r"% i)
    sys.stdout.flush()

f = open("test_result.txt", 'w')
print>>f, result3
print>>f, result4
f.close()

