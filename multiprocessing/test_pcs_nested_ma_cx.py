import numpy as np
import maest as ma
import cumxst as cx
import nested_maest as nma
import nested_cumxst as ncx
import impulse_response as ir
from multiprocessing import Process
from multiprocessing import Pool

def task_nma(pcs, taps, winsize, r, slicing):
    if len(taps) <= 3:
        file_tag = "short"
    else:
        file_tag = "long"

    f = open("../result/mns_montecarlo_%s_ma%d_%d_slice%d_%d.csv"%(file_tag, len(pcs), winsize, slicing, int(''.join(map(str,pcs)))), 'w')
    for i in range(r):
        signal = np.load("../data/exp_deviate_one_%d.npy"%(i))[:slicing]
        receive = ir.moving_average(taps, signal)
        temp = nma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)
        f.write('%s\n' % temp)
        print temp
    f.close()


def task_ncx(pcs, taps, winsize, r, slicing):
    if len(taps) <= 3:
        file_tag = "short"
    else:
        file_tag = "long"

    f = open("../result/mns_montecarlo_%s_cx%d_%d_%d_slice%d.csv"%(file_tag, len(pcs), winsize, int(''.join(map(str,pcs))), slicing), 'w')
    for i in range(r):
        signal = np.load("../data/exp_deviate_one_%d.npy"%(i))[:slicing]
        receive = ir.moving_average(taps, signal)
        temp = ncx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)
        f.write('%s\n' % temp)
        print temp
    f.close()


def task_ma(pcs, taps, winsize, r, slicing):
    if len(taps) <= 3:
        file_tag = "short"
    else:
        file_tag = "long"

    f = open("../result/pcs_montecarlo_%s_ma%d_%d_slice%d_%d.csv"%(file_tag, len(pcs), winsize, slicing, int(''.join(map(str,pcs)))), 'w')
    for i in range(r):
        signal = np.load("../data/exp_deviate_one_%d.npy"%(i))[:slicing]
        receive = ir.moving_average(taps, signal)
        temp = ma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)
        f.write('%s\n' % temp)
        print temp
    f.close()


def task_cx(pcs, taps, winsize, r, slicing):
    if len(taps) <= 3:
        file_tag = "short"
    else:
        file_tag = "long"

    f = open("../result/pcs_montecarlo_%s_cx%d_%d_%d_slice%d.csv"%(file_tag, len(pcs), winsize, int(''.join(map(str,pcs))), slicing), 'w')
    for i in range(r):
        signal = np.load("../data/exp_deviate_one_%d.npy"%(i))[:slicing]
        receive = ir.moving_average(taps, signal)
        temp = cx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)
        f.write('%s\n' % temp)
        print temp
    f.close()


def main():
    job = Pool(8)
    r = 50
    winsize = 512
    

    taps = [1, -2.333, 0.667]
    pcs = [2,3,5]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_cx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))

    pcs = [1,1,1]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_cx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [4,3,4]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_ncx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [2,3,5]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_ma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [1,1,1]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_ma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [4,3,4]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_nma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    taps = [1, 0.1, -1.87, 3.02, -1.435, 0.49]
    pcs = [2,3,5]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_cx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))

    pcs = [1,1,1]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_cx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [4,3,4]
    for slicing in range(10000,710000,10000):
        job.apply_async(task_ncx, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [2,3,5]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_ma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [1,1,1]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_ma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    
    pcs = [4,3,4]
    for slicing in range(100000,710000,100000):
        job.apply_async(task_nma, args=(pcs, taps, winsize, r, slicing))
        print "winsize %s for ma(%s)" % (winsize, len(pcs))
    

    job.close()
    job.join()

if __name__ == "__main__":
    main()
        

