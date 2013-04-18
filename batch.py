import numpy as np
from maest import maest
import sys

signal = np.load("data/exp_deviate_one_0.npz.npy")
y = np.zeros(len(signal)-2)
b1 = -2.333
b2 = 0.667
for i in range(len(y)):
    y[i] = signal[i+2]+b1*signal[i+1]+b2*signal[i]

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
for i in result3:
    print>>f, i
print>>f, "\n\n"
for i in result4:
    print>>f, i
f.close()

