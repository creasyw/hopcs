import numpy as np
import maest as ma
import impulse_response as ir
from multiprocessing import Process

def task(maorder, pcs, winsize, r):
  f = open("pcs_montecarlo_ma%d_%d_%d.csv"%(maorder, winsize, int(''.join(map(str,pcs)))), 'w')
  for i in range(r):
    signal = np.load("/home/q80022617/work/data/exp_deviate_one_%d.npy"%(i))
    receive = ir.moving_average(2, [1,-2.333, 0.677], signal)
    temp = ma.maestx(receive, pcs, 2, maorder, winsize)
    f.write('%s\n' % temp)
    print temp
  f.close()


def main():
  pcs = [1,1,2,3]
  threads = []
  #winsize = 256
  r = 50
  job = []
  for maorder in range(3,5):
    for i in range(7,12):
      winsize = 2**i
      job.append(Process(target=task, args=(maorder, pcs, winsize, r)))
      job[-1].start()
      print "winsize %s for ma(%s)" % (winsize, maorder)


if __name__ == "__main__":
  main()
    

