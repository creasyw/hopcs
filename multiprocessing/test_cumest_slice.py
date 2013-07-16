import numpy as np
import cumxst as cx
import impulse_response as ir
from multiprocessing import Pool

def task(pcs, taps, winsize, r, slice):
  if len(taps) <= 3:
    file_tag = "short"
  else:
    file_tag = "long"

  f = open("pcs_montecarlo_%s_ma%d_%d_%d_slice%d.csv"%(file_tag, len(pcs), winsize, int(''.join(map(str,pcs))), slice), 'w')
  for i in range(r):
    signal = np.load("/home/q80022617/work/data/exp_deviate_one_%d.npy"%(i))[:slice]
    receive = ir.moving_average(taps, signal)
    temp = cx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def main():
  pcs = [1,2,3]
  taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
  job = Pool(23)
  winsize = 512
  r = 50
  
  for slice in range(10000,710000,10000):
    job.apply_async(task, args=(pcs, taps, winsize, r, slice))
    print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  taps = [1, -2.333, 0.667]
  for slice in range(10000,710000,10000):
    job.apply_async(task, args=(pcs, taps, winsize, r, slice))
    print "winsize %s for ma(%s)" % (winsize, len(pcs))
  
  job.close()
  job.join()

if __name__ == "__main__":
  main()
    

