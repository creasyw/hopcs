import sys
import numpy as np
from math import ceil, log
from fractions import gcd
import matplotlib.pyplot as plt
from itertools import product

norm = lambda m: (reduce(lambda acc, itr: acc+itr**2, m, 0))**0.5

def estc2 (signal, pcs, cplst, delay, NFFT):
    """
    Return estimated 2nd order statistics (autocorrelation coefficients).
    Input:  pcs: the loaded PCS (in a dictionary).
            cplst: the list of pairwise coprime factors.
            Note that the elements index in cplst should be corresponding to the key in the pcs.
    """
    temp = {}
    length = delay*2+1
    rxx = np.zeros((length, 2))
    numlst = [int(ceil(NFFT/k)) for k in cplst]
    maxstep = int(ceil(len(signal)/float(NFFT)))
    result = np.zeros((maxstep, length))

    count = 0
    while True:
        if count == maxstep: break
        for j in range(len(cplst)):
            temp[j] = pcs[j][count*numlst[j]:(count+1)*numlst[j]]
        for k in range(len(temp)):
            for g in range(len(temp)):
                x1 = temp[k]
                x2 = temp[g]
                for i in range(len(x1)):
                    if x1[i][0] == 0: continue
                    for j in range(len(x2)):
                        if x2[j][0] == 0: continue
                        index = -x2[j][1]+x1[i][1]+delay
                        if index >= length or index < 0: continue
                        if rxx[index][1] == 0:
                            rxx[index][0] = x1[i][0]*x2[j][0]
                            rxx[index][1] += 1
                        else:
                            rxx[index][0] = (rxx[index][1]*rxx[index][0] + x1[i][0]*x2[j][0]) / (rxx[index][1]+1)
                            rxx[index][1] += 1
        result[count, :] = rxx[:,0]
        count += 1
    #np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))
    return result

def estc3 (signal, pcs, cplst, delay, NFFT):
    """
    Return estimated 2nd order statistics (autocorrelation coefficients).
    Input:  pcs: the loaded PCS (in a dictionary).
            cplst: the list of pairwise coprime factors.
            Note that the elements index in cplst should be corresponding to the key in the pcs.
    """
    temp = {}
    length = delay*2+1
    rxx = np.zeros((length, 2))
    numlst = [int(ceil(NFFT/k)) for k in cplst]
    maxstep = int(ceil(len(signal)/float(NFFT)))
    result = np.zeros((maxstep, length))

    count = 0
    while True:
        if count == maxstep: break
        for j in range(len(cplst)):
            temp[j] = pcs[j][count*numlst[j]:(count+1)*numlst[j]]
        for k in range(len(temp)):
            for g in range(len(temp)):
                x1 = temp[k]
                x2 = temp[g]
                for i in range(len(x1)):
                    if x1[i][0] == 0: continue
                    for j in range(len(x2)):
                        if x2[j][0] == 0: continue
                        index = -x2[j][1]+x1[i][1]+delay
                        if index >= length: continue
                        if index < 0: break
                        if rxx[index][1] == 0:
                            rxx[index][0] = x1[i][0]*(x2[j][0]**2)
                            rxx[index][1] += 1
                        else:
                            rxx[index][0] = (rxx[index][1]*rxx[index][0] + x1[i][0]*(x2[j][0]**2)) / (rxx[index][1]+1)
                            rxx[index][1] += 1
        result[count, :] = rxx[:,0]
        count += 1
    #np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))
    return result

def test_coprime (input):
    if len(input)<=1: return False
    for i in range(len(input)):
        if input[i]<=0: return False
        for j in range(i+1, len(input)):
            if gcd(input[i],input[j])!=1: return False
    return True

def mapping(coprime_pair):
    N = reduce(lambda x,y: x*y, coprime_pair)
    result = defaultdict(list)
    dict = {}
    signal = set()
    for i in coprime_pair:
        signal |= set([k for k in range(-N, N) if k%i==0])
    signal = sorted(list(signal))

    for pairs in product(signal, repeat=2):
        if abs(pairs[0]-pairs[1])<N:
            result[abs(pairs[0]-pairs[1])].append([pairs[0], pairs[1]])
    return result


def sampling (signal, NFFT, clist):
    """
    Return the co-prime sampled signals.
    The format of return values is hashtable with 2-d matrix for each entry:
        (index in original signal, signal value)
    Input:  signal values (1-d vector)
            NFFT: FFT length. constant.
            clist: corresponding to coprime_list in main()
    """
    steps = int(ceil(len(signal)/float(NFFT)))
    print "The length of signal is ", len(signal)
    print "...And it is splitted into %d pieces" % (steps)
    
    product = reduce(lambda x,y: x*y, clist)
    if product > NFFT:
        print "Warning: There might be blind spots in a FFT segmentation"

    sampled = {}
    for i in range(len(clist)):
        index = np.sort([j+k*NFFT for j in xrange(0,NFFT,clist[i]) for k in range(steps)])
        sampled[i] = np.array([[signal[j],j] for j in index])
        print "For this slot, the length of sampled sequence is ", len(sampled[i])
    return sampled

def main (signal, delay, round,  coprime_list):
    """
    Main loop for sampling and processing.
    The signalfile and pcsfile cannot be blank for the same time.
    The specific formats of them refer to the example in the bottom.
    NFFT: the length of FFT
    coprime_list: the list of pairwise coprime numbers.
    Note that: 1) the product for coprime_list should be smaller than NFFT.
               2) EITHER signalfile and pcsfile should be assigned.
    """
    # preprocessing
    NFFT = 512
    assert test_coprime(coprime_list), "The input coprime list is illegal."
    coprime_list.sort()
    pcs = sampling(signal, NFFT, coprime_list)
    c2 = estc2(signal, pcs, coprime_list, delay, NFFT)
    c3 = estc3(signal, pcs, coprime_list, delay, NFFT)
    delta = c2[:,-1]*c3[:,0]/c3[:,-1]
    b2 = c3[:,-1]/c3[:,0]
    b1 = c2[:,-2]/(delta*(1+delta)*b2)
    print "complete round ", round
    np.save("result/exp_deviate_b1_cp_%d.npy"%(round), b1)
    np.save("result/exp_deviate_b2_cp_%d.npy"%(round), b2)



def full_estc2(signal, delay):
    length = delay*2+1
    rxx = np.zeros((length, 2))
    maxstep = len(signal)/length
    result = np.zeros((maxstep, length))

    count = 0
    while True:
        if count == maxstep: break
        x = signal[count*length:(count+1)*length]
        for i in range(len(x)):
            if x[i] == 0: continue
            for j in range(len(x)):
                if x[j] == 0: continue
                index = i-j+delay
                if index >= length or index < 0: continue
                if rxx[index][1] == 0:
                    rxx[index][0] = x[i]*x[j]
                    rxx[index][1] += 1
                else:
                    rxx[index][0] = (rxx[index][1]*rxx[index][0] + x[i]*x[j]) / (rxx[index][1]+1)
                    rxx[index][1] += 1
        result[count, :] = rxx[:,0]
        count += 1
    #np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))
    return result

def full_estc3(signal, delay):
    length = delay*2+1
    rxx = np.zeros((length, 2))
    maxstep = len(signal)/length
    result = np.zeros((maxstep, length))

    count = 0
    while True:
        if count == maxstep: break
        x = signal[count*length:(count+1)*length]
        for i in range(len(x)):
            if x[i] == 0: continue
            for j in range(len(x)):
                if x[j] == 0: continue
                index = j-i+delay
                if index >= length or index < 0: continue
                if rxx[index][1] == 0:
                    rxx[index][0] = x[i]*(x[j]**2)
                    rxx[index][1] += 1
                else:
                    rxx[index][0] = (rxx[index][1]*rxx[index][0] + x[i]*(x[j]**2)) / (rxx[index][1]+1)
                    rxx[index][1] += 1
        result[count, :] = rxx[:,0]
        count += 1
    #np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))
    return result


def benchmark(output, delay, round):
    # only consider 2 delays
    c2 = full_estc2(output, delay)
    c3 = full_estc3(output, delay)
    delta = c2[:,-1]*c3[:,0]/c3[:,-1]
    b2 = c3[:,-1]/c3[:,0]
    b1 = c2[:,-2]/(delta*(1+delta)*b2)
    print "complete round ", round
    np.save("result/exp_deviate_b1_full_%d.npy"%(round), b1)
    np.save("result/exp_deviate_b2_full_%d.npy"%(round), b2)

if __name__ == "__main__":
    # dealing with new signal and meanwhile dumping PCS
#    main(512, [3,4,5,7], "data/exp_deviate_one.npy")
    
    # regular experiment processing PCS
#    filelist = ["data/pcs_data_3.npy", "data/pcs_data_4.npy", "data/pcs_data_5.npy", "data/pcs_data_7.npy"]

    # testing the benchmark
    nmc = 50
    for j in range(nmc):
        signal = np.load("data/exp_deviate_one_%d.npz.npy"%(j))
        # For 3,4,5,7 (420), using 512*100=51200 points (100 frames)
        y = np.zeros(102400)
        b1 = -2.333
        b2 = 0.667
        for i in range(len(y)):
            y[i] = signal[i+2]+b1*signal[i+1]+b2*signal[i]
#        benchmark(y, 2, j)
        main(y, 2, j, [3,4,5,7])
