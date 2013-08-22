import numpy as np
import maest as ma
import cumxst as cx
import impulse_response as ir
from multiprocessing import Process
from multiprocessing import Pool

def task_cx(pcs, taps, testing_order, winsize, r, slicing):
  f = open("cx_order_test_%s_hos%d_%d_slice%d_%d.csv"%(testing_order, len(pcs), winsize, slicing, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/creasy/workplace/data/exp_deviate_one_%d.npy"%(i))[:slicing]
    receive = ir.moving_average(taps, signal)
    temp = cx.cumx(receive, pcs, len(pcs), testing_order, winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def task_ma(pcs, taps, testing_order, winsize, r, slicing):
  f = open("ma_order_test_%s_hos%d_%d_slice%d_%d.csv"%(testing_order, len(pcs), winsize, slicing, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/creasy/workplace/data/exp_deviate_one_%d.npy"%(i))[:slicing]
    receive = ir.moving_average(taps, signal)
    temp = ma.maestx (receive, pcs, testing_order, len(pcs), winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()



def main():
  job = Pool(8)
  r = 50
  winsize = 512
  
  taps = [1, -2.333, 0.667]
  pcs = [1,2,3]
  # test order mismatch
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_ma, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs = [1,1,2,3]
  # test order mismatch
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_ma, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
  pcs = [1,2,3]
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_ma, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs = [1,1,2,3]
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_ma, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  taps = [1, -2.333, 0.667]
  pcs = [1,2,3]
  # test order mismatch
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_cx, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs = [1,1,2,3]
  # test order mismatch
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_cx, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
  pcs = [1,2,3]
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_cx, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  pcs = [1,1,2,3]
  for slicing in range(5000,200000,5000):
    for order in range(2, 9):
      job.apply_async(task_cx, args=(pcs, taps, order, winsize, r, slicing))
      print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  job.close()
  job.join()

if __name__ == "__main__":
  main()
    

