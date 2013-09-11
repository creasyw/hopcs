import numpy as np
from cumxst import cumx
from cumest import cumest
import impulse_response as ir
from arorder import ar_estimate

def arma(pcs, ar, ma, winsize, mc_round, slicing, snr, noise_type):
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
        f1 = open("result/ar%s_hos%d_winsize%d_slice%d_snr%d_pcs%s.csv"%(ar, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        f2 = open("result/ma%s_hos%d_winsize%d_slice%d_snr%d_pcs%s.csv"%(ma, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        for i in range(mc_round):
            receive = np.load("temp/arma_data_%d.npy"%(i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f1.write('%s\n' % temp)
            print "AR - snr=+inf, ", temp
            temp = cumx(receive, pcs, len(pcs), ma, winsize)
            f2.write('%s\n' % temp)
            print "MA - snr=+inf, ", temp

    elif noise_type=="white":
        f1 = open("result/ar%s_hos%d_winsize%d_slice%d_white_snr%d_pcs%s.csv"%(ar, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        f2 = open("result/ma%s_hos%d_winsize%d_slice%d_white_snr%d_pcs%s.csv"%(ma, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        for i in range(mc_round):
            receive = np.load("temp/arma_data_white_%d_%d.npy"%(snr, i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f1.write('%s\n' % temp)
            print "AR - white noise, snr=%d, "%(snr), temp
            temp = cumx(receive, pcs, len(pcs), ma, winsize)
            f2.write('%s\n' % temp)
            print "MA - white noise, snr=%d, "%(snr), temp

    elif noise_type=="color":
        f1 = open("result/ar%s_hos%d_winsize%d_slice%d_color_snr%d_pcs%s.csv"%(ar, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        f2 = open("result/ma%s_hos%d_winsize%d_slice%d_color_snr%d_pcs%s.csv"%(ma, len(pcs), winsize, slicing, snr, ''.join([str(k) for k in pcs]) ), 'w')
        for i in range(mc_round):
            receive = np.load("temp/arma_data_color_%d_%d.npy"%(snr, i))[:slicing]
            temp = ar_estimate(receive, pcs, ar, ma, winsize)
            f1.write('%s\n' % temp)
            print "AR - color noise, snr=%d, "%(snr), temp
            temp = cumx(receive, pcs, len(pcs), ma, winsize)
            f2.write('%s\n' % temp)
            print "MA - color noise, snr=%d, "%(snr), temp

    else:
        print "ERROR: the noise type is wrong!!"
    
    f1.close()
    f2.close()

