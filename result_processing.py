#! /usr/bin/python

import re
import sys, os

def avg(lst):
  return sum(lst)/float(len(lst))

def var(lst):
  lavg = avg(lst)
  diff = [k-lavg for k in lst]
  return sum(diff)/float(len(diff))


rate = re.compile("Avg User Rate  = (\d+.\d*)")
utility = re.compile("Utility = (\d+.\d*)")

filename = sys.argv[-1]
rt_list = []
ut_list = []
with open(os.path.join(os.path.dirname(__file__), filename)) as datafile:
  for row in datafile:
    hit = rate.findall(row)
    if len(hit) != 0:
      rt_list.append(float(hit[0]))
    hit = utility.findall(row)
    if len(hit) != 0:
      ut_list.append(float(hit[0]))


print "Rate avg: ", avg(rt_list), ", variance: ", var(rt_list), \
    ", max: ", max(rt_list), ", min: ", min(rt_list)
print "Utility avg: ", avg(ut_list), ", variance: ", var(ut_list), \
    ", max: ", max(ut_list), ", min: ", min(ut_list)


