import numpy as np
import maest as ma
import impulse_response as ir
from multiprocessing import Process

def task(maorder, pcs, taps, winsize, r):
  f = open("pcs_montecarlo_long_ma%d_%d_%d.csv"%(maorder, winsize, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/q80022617/work/data/exp_deviate_one_%d.npy"%(i))
    receive = ir.moving_average(taps, signal)
    temp = ma.maestx(receive, pcs, 2, maorder, winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def main():
  pcs_lst = [[1,1,2,3], [1,1,2,5], [1,1,2,7], [1,1,3,5]]
  taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
  threads = []
  #winsize = 256
  maorder = 4
  r = 50
  job = []
  for pcs in pcs_lst:
    for i in range(7,12):
      winsize = 2**i
      job.append(Process(target=task, args=(maorder, pcs, taps, winsize, r)))
      job[-1].start()
      print "winsize %s for ma(%s)" % (winsize, maorder)


if __name__ == "__main__":
  main()
    

