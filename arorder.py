import numpy as np
from cumxst import cumx
from cumest import cumest
import impulse_response as ir

def ar_estimate(sig, pcs, ar, ma, winsize):
    if len(pcs) != 3:
        raise ValueError("The ar estimate could only handle 3rd-order cumulant")
    rb = ma+ar+1
    m = np.zeros((rb, 2*ar+ma))
    for p in range(rb):
        temp = cumx(sig, pcs, 3, 2*ar+ma-1, winsize, 0, p-ar)
        m[p,0] = temp[len(temp)/2]
        #m[p,1:] = (temp[:len(temp)/2][::-1]+temp[len(temp)/2+1:])/2
        m[p,1:] = temp[len(temp)/2+1:]
    m = m.T
    # put cumulants into Hankel matrix
    result = np.zeros((ar*rb, ar))
    for i in range(ar*rb):
        for j in range(ar):
            result[i,j] = m[ma+j+1+i/rb, ar-i%rb]
    _, s, _ = np.linalg.svd(result)
    return s/s[0]

def pcs_ar(pcs, ar, ma, winsize, mc_round, slicing, snr, noise_type):
    """
    Input variables:
        pcs: array-like pcs choices
        ar, ma: order for the ar and ma models
        winsize: length of processing segmentation
        mc_round: number of monte carlo simulations
        slicing: length of slicing for the generated data
        snr: snr level for specific data file (larger than 100 for noise free
        noice_type: white or color.
    """
    if snr > 100:
        f = open("result/ar_testorder%s_hos%d_winsize%d_slice%d_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(mc_round):
            receive = np.load("temp/ar_data_%d.npy"%(i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f.write('%s\n' % temp)
            print "snr=+inf, ", temp
    elif noise_type=="white":
        f = open("result/ar_testorder%s_hos%d_winsize%d_slice%d_white_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(mc_round):
            receive = np.load("temp/ar_data_white_%d_%d.npy"%(snr, i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f.write('%s\n' % temp)
            print "white noise, snr=%d, "%(snr), temp
    elif noise_type=="color":
        f = open("result/ar_testorder%s_hos%d_winsize%d_slice%d_color_snr%d.csv"%(testing_order, len(pcs), winsize, slicing, snr), 'w')
        for i in range(mc_round):
            receive = np.load("temp/ar_data_color_%d_%d.npy"%(snr, i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f.write('%s\n' % temp)
            print "color noise, snr=%d, "%(snr), temp
    else:
        print "ERROR: the noise type is wrong!!"
    f.close()


