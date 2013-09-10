import numpy as np
from cumxst import cumx
from cumest import cumest
import impulse_response as ir

def pcs_cx(pcs, testing_order, winsize, r, slicing, snr, noise_type):
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

def hos_cm(hos_order, testing_order, winsize, r, slicing, snr, noise_type):
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


