#! /usr/bin/python

import re
import sys, os
import numpy as np

def avg(lst):
  return sum(lst)/float(len(lst))

def var(lst):
  lavg = avg(lst)
  diff = [k-lavg for k in lst]
  return sum(diff)/float(len(diff))

finder = re.compile("-?\d+\.\d*[[eE]?[-+]?\d*")

filename = sys.argv[-1]
result = []
with open(os.path.join(os.path.dirname(__file__), filename)) as datafile:
  for row in datafile:
    hit = finder.findall(row)
    if len(hit) != 0:
      result.append(map(lambda x: float(x), hit))

result = np.array(result)

# post-processing
tap1 = result[:,1]
tap2 = result[:,2]
threshold = 20

mtap1 = np.median(tap1)
mtap2 = np.median(tap2)
tap1 = filter(lambda x: abs(x-mtap1)<10, tap1)
tap2 = filter(lambda x: abs(x-mtap2)<10, tap2)
print len(tap1), len(tap2)
print filename, np.mean(tap1), np.var(tap1), np.mean(tap2), np.var(tap2)




