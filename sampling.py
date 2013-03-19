import numpy as np
from math import ceil, log
from fractions import gcd
import matplotlib.pyplot as plt

def autocorrelation (r_xx, x1, x2, zero):
    """
    Calculate the autocorrelation operating upon the r_xx.
    Return the modified array of autocorrelation.
    Input:  r_xx: previous autocorrelation.
            x1, x2: co-prime sampled signal.
            zero: relative zero point along with moving window.
    """
    for i in range(len(x1)):
        if x1[i][1] == 0:
            continue
        for j in range(len(x2)):
            if x2[j][1] == 0:
                continue
            index = abs(x1[i][0]-x2[j][0])
            #if index > 255:
            #    continue
            if zero+index < len(r_xx):
                index += zero
                if r_xx[index][1] == 0:
                    r_xx[index][0] = x1[i][1]*(x2[j][1].conj())
                    r_xx[index][1] += 1
                else:
                    r_xx[index][0] = (r_xx[index][1]*r_xx[index][0] + x1[i][1]*(x2[j][1].conj())) / (r_xx[index][1]+1)
                    r_xx[index][1] += 1
    return r_xx

def dft (r_xx, Fs, NFFT, hamming, overlap=True, sides='default'):
    if overlap:
        step = NFFT // 2
    else:
        step = NFFT
    ind = np.arange(0, len(r_xx)-NFFT+ 1, step)
    n = len(ind)
    pad_to = NFFT
    if (sides == 'default' and np.iscomplexobj(r_xx)) or sides == 'twosided':
        numFreqs = pad_to
        scaling_factor = 1.
    elif sides in ('default', 'onesided'):
        numFreqs = pad_to//2 + 1
        scaling_factor = 2.
    else:
        raise ValueError("sides must be one of: 'default', 'onesided', or 'twosided'")
    
    psd = np.zeros((numFreqs, n), np.complex_)
    
    for i in range(n):
        temp = r_xx[ind[i]:(ind[i]+NFFT),0]*hamming
        psd[:,i] = np.fft.fft(temp, n=pad_to)[:numFreqs]
        #psd[:,i] = np.conjugate(fx[:numFreqs])*fx[:numFreqs]
    
    # Also include scaling factors for one-sided densities and dividing by the
    # sampling frequency, if desired. Scale everything, except the DC component
    # and the NFFT/2 component:
    psd[1:-1] *= scaling_factor

    # MATLAB divides by the sampling frequency so that density function
    # has units of dB/Hz and can be integrated by the plotted frequency
    # values. Perform the same scaling here.
    psd /= Fs
    
    t = 1./Fs * (ind + NFFT/2.)
    freqs = float(Fs) / pad_to * np.arange(numFreqs)

    if (np.iscomplexobj(r_xx) and sides == 'default') or sides == 'twosided':
        # center the frequency range at zero
        freqs = np.concatenate((freqs[numFreqs//2:] - Fs, freqs[:numFreqs//2]))
        psd = np.concatenate((psd[numFreqs//2:, :], psd[:numFreqs//2, :]), 0)

    return psd, freqs, t

def visualize_rxx (r_xx, Fs, NFFT):
    step = NFFT
    ind = np.arange(0, len(r_xx)-NFFT+ 1, step)
    n = len(ind)
    pad_to = NFFT
    numFreqs = NFFT//2+1
    psd = np.zeros((numFreqs, n), np.complex_)
    t = 1./Fs * (ind + NFFT/2.)
    freqs = float(Fs) / pad_to * np.arange(numFreqs)
    for i in range(n):
        psd[:,i] = r_xx[ind[i]:(ind[i]+NFFT),0][:numFreqs]
    return psd, freqs, t

def test_coprime (input):
    if len(input)<=1: return False
    for i in range(len(input)):
        if input[i]<=0: return False
        for j in range(i+1, len(input)):
            if gcd(input[i],input[j])!=1: return False
    return True

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

def main (NFFT, coprime_list, signalfile='', pcsfile=''):
    """
    Main loop for sampling and processing.
    The signalfile and pcsfile cannot be blank for the same time.
    The specific formats of them refer to the example in the bottom.
    NFFT: the length of FFT
    coprime_list: the list of pairwise coprime numbers.
    Note that the product for coprime_list should be smaller than NFFT.
    """
    # preprocessing
    assert test_coprime(coprime_list), "The input coprime list is illegal."
    coprime_list.sort()
    if signalfile == '':
        pcs = {}
        counter = 0
        for filename in pcsfile:
            pcs[counter] = np.load(filename)
            print "For this slot, the length of sampled sequence is ", len(pcs[counter])
            counter += 1
    else:
        signal = np.load(signalfile)
        pcs = sampling(signal, NFFT, coprime_list)
        for i in range(len(pcs)):
            np.save("pcs_data_%d.npy"%(coprime_list[i]), pcs[i])
    print "everthing done"

if __name__ == "__main__":
    # dealing with new signal and meanwhile dumping PCS
#    main(512, [3,4,5,7], "data/exp_deviate_one.npy")
    # regular experiment processing PCS
    filelist = ["data/pcs_data_3.npy", "data/pcs_data_4.npy", "data/pcs_data_5.npy", "data/pcs_data_7.npy"]
    main(512, [3,4,5,7], pcsfile = filelist)
