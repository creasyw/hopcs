import sys
import numpy as np
from math import ceil, log
from fractions import gcd
import matplotlib.pyplot as plt

norm = lambda m: (reduce(lambda acc, itr: acc+itr**2, m, 0))**0.5

def estc2 (pcs, cplst, NFFT):
    """
    Return estimated 2nd order statistics (autocorrelation coefficients).
    Input:  pcs: the loaded PCS (in a dictionary).
            cplst: the list of pairwise coprime factors.
            Note that the elements index in cplst should be corresponding to the key in the pcs.
    """
    temp = {}
    threshold = 0.1
    numlst = [int(ceil(NFFT/k)) for k in cplst]
    rxx = np.zeros((NFFT, 2))
    # all four elements in cplist should have the same # of splits
    # concerning len(pcs[i]/numlst[i]
    maxstep = len(pcs[0])/numlst[0]
    count = 0
    while True:
        if count == maxstep: break
        prevrxx = np.array(rxx)
        for j in range(len(cplst)):
            temp[j] = pcs[j][count*numlst[j]:(count+1)*numlst[j]]
        x1 = temp[0]
        x2 = temp[1]
        for i in range(len(x1)):
            if x1[i][0] == 0:
                continue
            for j in range(len(x2)):
                if x2[j][0] == 0: continue
                index = abs(x1[i][1]-x2[j][1])
                if index >= NFFT: continue
                if rxx[index][1] == 0:
                    rxx[index][0] = x1[i][0]*(x2[j][0].conj())
                    rxx[index][1] += 1
                else:
                    rxx[index][0] = (rxx[index][1]*rxx[index][0] + x1[i][0]*(x2[j][0].conj())) / (rxx[index][1]+1)
                    rxx[index][1] += 1
        #rxx = (count*prevrxx+rxx)/(count+1)
        count += 1
        if norm((prevrxx-rxx)[:,0]) <= threshold: print count; break
        #if count == 100: break
        #print count, rxx[:10]
        #print "\n", norm((prevrxx-rxx)[:,0])

def estc3 (pcs, cplst, NFFT):
    """
    Return estimated 2nd order statistics (autocorrelation coefficients).
    Input:  pcs: the loaded PCS (in a dictionary).
            cplst: the list of pairwise coprime factors.
            Note that the elements index in cplst should be corresponding to the key in the pcs.
    """
    temp = {}
    threshold = 0.1
    numlst = [int(ceil(NFFT/k)) for k in cplst]
    rxx = np.zeros((NFFT, 2))
    # all four elements in cplist should have the same # of splits
    # concerning len(pcs[i]/numlst[i]
    maxstep = len(pcs[0])/numlst[0]
    count = 0
    while True:
        prevrxx = np.array(rxx)
        for j in range(len(cplst)):
            temp[j] = pcs[j][count*numlst[j]:(count+1)*numlst[j]]
        x1 = temp[0]
        x2 = temp[1]
        x3 = temp[3]
        for i in range(len(x1)):
            if x1[i][0] == 0:
                continue
            for j in range(len(x2)):
                if x2[j][0] == 0: continue
                index = abs(x1[i][1]-x2[j][1])
                if index >= NFFT: continue
                if rxx[index][1] == 0:
                    rxx[index][0] = x1[i][0]*(x2[j][0].conj())
                    rxx[index][1] += 1
                else:
                    rxx[index][0] = (rxx[index][1]*rxx[index][0] + x1[i][0]*(x2[j][0].conj())) / (rxx[index][1]+1)
                    rxx[index][1] += 1
        #rxx = (count*prevrxx+rxx)/(count+1)
        count += 1
        if norm((prevrxx-rxx)[:,0]) <= threshold: print count; break
        #if count == 100: break
        #print count, rxx[:10]
        #print "\n", norm((prevrxx-rxx)[:,0])

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
    Note that: 1) the product for coprime_list should be smaller than NFFT.
               2) EITHER signalfile and pcsfile should be assigned.
    """
    # preprocessing
    assert test_coprime(coprime_list), "The input coprime list is illegal."
    coprime_list.sort()
    # Load existing PCS datafile from pcsfiles
    if signalfile == '':
        pcs = {}
        counter = 0
        for filename in pcsfile:
            pcs[counter] = np.load(filename)
            print "For this slot, the length of sampled sequence is ", len(pcs[counter])
            counter += 1
        #TODO: sanity check for coprime pairs and the loaded PCS
    # without downsampling, only perform coprime sampling and dumping datafile
    else:
        signal = np.load(signalfile)
        pcs = sampling(signal, NFFT, coprime_list)
        for i in range(len(pcs)):
            np.save("pcs_data_%d.npy"%(coprime_list[i]), pcs[i])
    
    # loading complete
    estc2(pcs, coprime_list, NFFT)



def full_estc2(NFFT, signal, round):
    """ round: the # of monte carlo simulation """
    rxx = np.zeros((NFFT, 2))
    # For current setting of simulation the 512-FFT window will move 1386 times
#    maxstep = len(signal)/NFFT
    # For the sake of saving time of computation, hard coded as 100
    maxstep = 100

    count = 0
    estc2full = []
    while True:
        if count == maxstep: break
        x = signal[count*NFFT:(count+1)*NFFT]
        prevrxx = np.array(rxx)
        for i in range(len(x)):
            if x[i] == 0: continue
            for j in range(len(x)):
                if x[j] == 0: continue
                index = abs(i-j)
                if index >= NFFT: continue
                if rxx[index][1] == 0:
                    rxx[index][0] = x[i]*(x[j].conj())
                    rxx[index][1] += 1
                else:
                    rxx[index][0] = (rxx[index][1]*rxx[index][0] + x[i]*(x[j].conj())) / (rxx[index][1]+1)
                    rxx[index][1] += 1
        count += 1
        estc2full.append(norm((prevrxx-rxx)[:,0]))
        sys.stdout.write("%s\r"%(count))
        sys.stdout.flush()
    np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))

def full_estc3(NFFT, signal, round):
    """ impractical to use... """
    rxx = np.zeros((NFFT, NFFT, 2))
    # For current setting of simulation the 512-FFT window will move 1386 times
#    maxstep = len(signal)/NFFT
    # For the sake of saving time of computation, hard coded as 100
    maxstep = 100

    count = 0
    estc3full = []
    while True:
        if count == maxstep: break
        x = signal[count*NFFT:(count+1)*NFFT]
        prevrxx = np.array(rxx)
        for i in range(len(x)):
            if x[i] == 0: continue
            for j in range(len(x)):
                if x[j] == 0: continue
                for g in range(len(x)):
                    if x[g] == 0: continue
                    index1 = abs(i-j)
                    index2 = abs(i-g)
                    if index1 >= NFFT or index2 >= NFFT: continue
                    if rxx[index1][index2][1] == 0:
                        rxx[index1][index2][0] = x[i]*(x[j].conj())*x[g]
                        rxx[index1][index2][1] += 1
                    else:
                        rxx[index1][index2][0] = (rxx[index1][index2][1]*rxx[index1][index2][0] + x[i]*(x[j].conj())*x[g]) / (rxx[index1][index2][1]+1)
                        rxx[index1][index2][1] += 1
        count += 1
        diff = (prevrxx-rxx)[:,:,0]
        estc3full.append(sum((diff.reshape(1, diff.shape[0]*diff.shape[1]))**2)**0.5)
        print estc3full[-1]
        #sys.stdout.write("%s\r"%(count))
        #sys.stdout.flush()
        if count == 10: break
    #np.save("result/exp_deviate_estc2_full_%d.npy"%(round), np.array(estc2full))


def benchmark(NFFT):
    nmc = 50
    for i in range(50):
        signal = np.load("data/exp_deviate_one_%d.npz.npy"%(i))
        full_estc2(NFFT, signal, i)
        print "Completed the round ", i

if __name__ == "__main__":
    # dealing with new signal and meanwhile dumping PCS
#    main(512, [3,4,5,7], "data/exp_deviate_one.npy")
    
    # regular experiment processing PCS
#    filelist = ["data/pcs_data_3.npy", "data/pcs_data_4.npy", "data/pcs_data_5.npy", "data/pcs_data_7.npy"]
#    main(512, [3,4,5,7], pcsfile = filelist)

    # testing the benchmark
    benchmark(512)
