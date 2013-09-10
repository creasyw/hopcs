import numpy as np
from cumxst import cumx
from cumest import cumest
import impulse_response as ir
from multiprocessing import Process
from multiprocessing import Pool


def ar_estimate(sig, pcs, ar, ma, winsize):
    if len(pcs) != 3:
        raise ValueError("The ar estimate could only handle 3rd-order cumulant")
    m = np.zeros((ar+1, 2*ar))
    for p in range(ar+1):
        temp = cumx(sig, pcs, 3, 2*ar-1, winsize, 0, -1*p)
        m[p,0] = temp[len(temp)/2]
        m[p,1:] = (temp[:len(temp)/2][::-1]+temp[len(temp)/2+1:])/2
    m = m.T
    # put the cumulants into algo. matrix
    rb = ma+ar+1
    result = np.zeros((ar*rb, ar))
    for i in range(ar*rb):
        for j in range(ar):
            result[i,j] = m[ma+j+1+i/rb, ar-i%rb]

def task_cx(pcs, testing_order, winsize, r, slicing, snr, noise_type):
    if snr > 100:
        f = open("result/cx_testorder%s_hos%d_winsize%d_slice%d_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_%d.npy"%(i))[:slicing]
            temp = cumx(receive, pcs, len(pcs), testing_order, winsize)
            f.write('%s\n' % temp)
            print "snr=+inf, ", temp
    elif noise_type=="white":
        f = open("result/cx_testorder%s_hos%d_winsize%d_slice%d_white_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_white_%d_%d.npy"%(snr, i))[:slicing]
            temp = cumx(receive, pcs, len(pcs), testing_order, winsize)
            f.write('%s\n' % temp)
            print "white noise, snr=%d, "%(snr), temp
    elif noise_type=="color":
        f = open("result/cx_testorder%s_hos%d_winsize%d_slice%d_color_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_color_%d_%d.npy"%(snr, i))[:slicing]
            temp = cumx(receive, pcs, len(pcs), testing_order, winsize)
            f.write('%s\n' % temp)
            print "color noise, snr=%d, "%(snr), temp
    else:
        print "ERROR: the noise type is wrong!!"
    f.close()

def task_cm(hos_order, testing_order, winsize, r, slicing, snr, noise_type):
    if snr > 100:
        f = open("result/cm_testorder%s_hos%d_winsize%d_slice%d_snr%d.csv"%(testing_order, hos_order, winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_%d.npy"%(i))[:slicing]
            temp = cumest(receive, hos_order, testing_order, winsize)
            f.write('%s\n' % temp)
            print "snr=+inf, ", temp
    elif noise_type=="white":
        f = open("result/cm_testorder%s_hos%d_winsize%d_slice%d_white_snr%d.csv"%(testing_order, hos_order, winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_white_%d_%d.npy"%(snr, i))[:slicing]
            temp = cumest(receive, hos_order, testing_order, winsize)
            f.write('%s\n' % temp)
            print "white noise, snr=%d, "%(snr), temp
    elif noise_type=="color":
        f = open("result/cm_testorder%s_hos%d_winsize%d_slice%d_color_snr%d.csv"%(testing_order, hos_order, winsize, slicing, snr), 'w')
        for i in range(r):
            receive = np.load("temp/data_color_%d_%d.npy"%(snr, i))[:slicing]
            temp = cumest(receive, hos_order, testing_order, winsize)
            f.write('%s\n' % temp)
            print "color noise, snr=%d, "%(snr), temp
    else:
        print "ERROR: the noise type is wrong!!"
    f.close()


def main():
    job = Pool(8)
    r = 50
    winsize = 512

    # snr = +inf
    pcs = [1,2,3]
    order = 6
    for slicing in range(5000,50001,5000):
        job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, 500, "white"))
        for snr in range(-10,21):
            job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, snr, "white"))
            job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, snr, "color"))
    pcs = [1,1,2,3]
    for slicing in range(5000,50001,5000):
        job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, 500, "white"))
        for snr in range(-10,21):
            job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, snr, "white"))
            job.apply_async(task_cx, args=(pcs, order, winsize, r, slicing, snr, "color"))
    #benchmark for non-PCS
    for slicing in range(5000,50001,5000):
        job.apply_async(task_cm, args=(2, order, winsize, r, slicing, 500, "white"))
        job.apply_async(task_cm, args=(3, order, winsize, r, slicing, 500, "white"))
        job.apply_async(task_cm, args=(4, order, winsize, r, slicing, 500, "white"))
        for snr in range(-10,21):
            job.apply_async(task_cm, args=(2, order, winsize, r, slicing, snr, "white"))
            job.apply_async(task_cm, args=(3, order, winsize, r, slicing, snr, "white"))
            job.apply_async(task_cm, args=(4, order, winsize, r, slicing, snr, "white"))
            job.apply_async(task_cm, args=(2, order, winsize, r, slicing, snr, "color"))
            job.apply_async(task_cm, args=(3, order, winsize, r, slicing, snr, "color"))
            job.apply_async(task_cm, args=(4, order, winsize, r, slicing, snr, "color"))


    job.close()
    job.join()

if __name__ == "__main__":
    main()


