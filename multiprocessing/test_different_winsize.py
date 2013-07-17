import numpy as np
import maest as ma
import maest_com as ma_com
import impulse_response as ir
from multiprocessing import Process
from multiprocessing import Pool

def task(pcs, taps, winsize, r):
  if len(taps) <= 3:
    file_tag = "short"
  else:
    file_tag = "long"

  f = open("pcs_montecarlo_%s_ma%d_%d_%d.csv"%(file_tag, len(pcs), winsize, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/q80022617/work/data/exp_deviate_one_%d.npy"%(i))
    receive = ir.moving_average(taps, signal)
    temp = ma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def task_com(pcs, taps, winsize, r):
  if len(taps) <= 3:
    file_tag = "short"
  else:
    file_tag = "long"

  f = open("pcs_montecarlo_%s_ma%d_%d_%d.csv"%(file_tag, len(pcs), winsize, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/q80022617/work/data/exp_deviate_one_%d.npy"%(i))
    receive = ir.moving_average(taps, signal)
    temp = ma_com.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def main():
  job = Pool(23)
  r = 50

  taps = [1, -2.333, 0.667]
  pcs_lst = [[1,1,2,3], [1,1,2,5], [1,1,2,7], [1,1,3,5]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[1,2,1,3], [1,2,1,5], [1,2,1,7], [1,3,1,5]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[1,3,1,2], [1,5,1,2], [1,7,1,2], [1,5,1,3]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[1,2,3], [1,2,5], [1,2,7], [1,3,5]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[1,3,2], [1,5,2], [1,7,2], [1,5,3]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[3,1,2], [5,1,2], [7,1,2], [5,1,3]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
  pcs_lst = [[1,1,2,3], [1,1,2,5], [1,1,2,7], [1,1,3,5]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task_com, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs_lst = [[1,2,3], [1,2,5], [1,2,7], [1,3,5]]
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.apply_async(task_com, args=(pcs, taps, winsize, r))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  job.close()
  job.join()

if __name__ == "__main__":
  main()
    

