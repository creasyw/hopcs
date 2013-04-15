import numpy as np

def cum2est (signal, maxlag, nsamp, overlap, flag):
    """
    CUM2EST Covariance function.
         y: input data vector (column)
         maxlag: maximum lag to be computed
         samp_seg: samples per segment (<=0 means no segmentation)
         overlap: percentage overlap of segments
         flag: 'biased', biased estimates are computed
               'unbiased', unbiased estimates are computed.
         y_cum: estimated covariance,
                C2(m)  -maxlag <= m <= maxlag
    """
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecord = (len(signal)-overlap)/(nsamp-overlap)

    y_cum = np.zeros(maxlag+1, dtype=float)
    ind = 0

    for i in range(nrecord):
        x = signal[ind:(ind+nsamp)]
        x = x-float(sum(x))/len(x)
        for k in range(maxlag+1):
            y_cum[k] = y_cum[k] + reduce(lambda m,n:m+n, x[:(nsamp-k)]*x[k:nsamp], 0)
        ind += nadvance
    if flag == "biased":
        y_cum = y_cum / (nsamp*nrecord)
    elif flag == "unbiased":
        y_cum = y_cum / (nrecord * (nsamp-np.array(range(maxlag+1))))
    else:
        raise Exception("The flag should be either 'biased' or 'unbiased'!!")
    if maxlag>0:
        y_cum = np.hstack((np.conjugate(y_cum[maxlag+1:0:-1]), y_cum))
    return y_cum


def cum3est (signal, maxlag, nsamp, overlap, flag, k1):
    """
    CUM3EST Third-order cumulants.
        y: input data vector (column)
        maxlag: maximum lag to be computed
        samp_seg: samples per segment
        overlap: percentage overlap of segments
        flag : 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.
        k1: the fixed lag in c3(m,k1): see below
        y_cum:  estimated third-order cumulant,
                 C3(m,k1)  -maxlag <= m <= maxlag
    """
    minlag = -maxlag
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecord = (len(signal)-overlap)/(nsamp-overlap)

    y_cum = np.zeros(maxlag*2+1, dtype=float)
    ind = 0
    nlags = 2*maxlag + 1

    if flag == "biased":
        scale = np.ones(nlags, dtype=float)/nsamp
    elif flag == "unbiased":
        lsamp = nsamp - abs(k1)
        scale = range(lsamp-maxlag,lsamp+1) + range(lsamp-1, lsamp-maxlag-1, -1)
        scale = np.ones(len(scale), dtype=float)/scale
    else:
        raise Exception("The flag should be either 'biased' or 'unbiased'!!")

    for i in range(nrecord):
        x = signal[ind:(ind+nsamp)]
        x = x-float(sum(x))/len(x)
        cx = np.conjugate(x)
        z = x*0
        if k1 >= 0:
            z[:nsamp-k1] = x[:nsamp-k1]*cx[k1:nsamp]
        else:
            z[-k1:nsamp] = x[-k1:nsamp]*cx[:nsamp+k1]

        y_cum[maxlag] = y_cum[maxlag] +  reduce(lambda m,n:m+n, z*x, 0)

        for k in range(1, maxlag+1):
            y_cum[maxlag-k] = y_cum[maxlag-k] + reduce(lambda m,n:m+n, z[k:nsamp]*x[:nsamp-k], 0)
            y_cum[maxlag+k] = y_cum[maxlag+k] + reduce(lambda m,n:m+n, z[:nsamp-k]*x[k:nsamp], 0)
        ind += nadvance

    return y_cum*scale/nrecord

def test ():
    # for tesating purpose
    # The right results are:
    #           "biased": [-0.12250513  0.35963613  1.00586945  0.35963613 -0.12250513]
    #           "unbiaed": [-0.12444965  0.36246791  1.00586945  0.36246791 -0.12444965]
    import scipy.io as sio
    y = sio.loadmat("matfile/demo/ma1.mat")['y']
    #print cum2est(y, 2, 128, 0, 'unbiased')

    # For the 3rd cumulant:
    #           "biased": [-0.18203039  0.07751503  0.67113035  0.729953    0.07751503]
    #           "unbiased": [-0.18639911  0.07874543  0.67641484  0.74153955  0.07937539]
    print cum3est(y, 2, 128, 0, 'unbiased', 1)

if __name__=="__main__":
    test()


