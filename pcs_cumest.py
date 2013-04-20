import numpy as np

def sampling (signal, factor):
    """
    Return signals with sampling period given with "factor", and stuff zeros in the interval.
    The format of return values is np.ndarray.
    NOTE: it is different from the same function in "sampling.py".
    """
    return np.array([signal[k] if k%3==0 else 0 for k in range(len(signal))])

def cum2x (x,y, maxlag, nsamp, overlap, flag):
    assert len(x) == len(y), "The two signal should be same length!"
    assert maxlag >= 0, " 'maxlag' must be non-negative!"
    if nsamp > len(x) or nsamp <= 0:
        nsamp = len(x)
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecs  = (len(x)-overlap)/nadvance
    nlags = 2*maxlag+1
    y_cum = np.zeros(nlags, dtype=float)
    count = np.zeros(nlags, dtype=float)

    ind = 0
    for k in range(nrecs):
        xs = x[ind:(ind+nsamp)]
        xs = xs - float(sum(xs))/len(xs)
        ys = y[ind:(ind+nsamp)]
        ys = ys - float(sum(ys))/len(ys)
        temp = xs*ys
        y_cum[maxlag] += reduce(lambda m,n:m+n,temp, 0)
        count[maxlag] += len(filter(lambda i: i>=0.01, temp))
        for m in range(1,maxlag+1):
            temp = xs[m:nsamp]*ys[:nsamp-m]
            y_cum[maxlag-m] = y_cum[maxlag-m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag-m] += len(filter(lambda i: i>=0.005, temp))
            temp = xs[:nsamp-m]*ys[m:nsamp]
            y_cum[maxlag+m] = y_cum[maxlag+m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag+m] += len(filter(lambda i: i>=0.005, temp))
        ind += nadvance
    if flag == "biased":
        scale = np.ones(nlags, dtype=float)/nsamp/nrecs
    elif flag == "unbiased":
        scale = 1./count
    else:
        raise Exception("The flag should be either 'biased' or 'unbiased'!!")
    return y_cum*scale


def cum2est (signal, maxlag, nsamp, overlap, flag):
    """
    CUM2EST Covariance function.
    For the case of PCS, only the nonzero sampled points are taken into account.
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
    count = np.zeros(maxlag+1, dtype=float)
    ind = 0

    for i in range(nrecord):
        x = signal[ind:(ind+nsamp)]
        x = x-float(sum(x))/len(x)
        for k in range(maxlag+1):
            y_cum[k] = y_cum[k] + reduce(lambda m,n:m+n, x[:(nsamp-k)]*x[k:nsamp], 0)
            count[k] += len(filter(lambda i: i!=0, x[k:nsamp]))
        ind += nadvance
    if flag == "biased":
        y_cum = y_cum / (nsamp*nrecord)
    elif flag == "unbiased":
        y_cum = y_cum / count
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
        scale = np.array(range(lsamp-maxlag,lsamp+1) + range(lsamp-1, lsamp-maxlag-1, -1))
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

def cum4est (signal, maxlag, nsamp, overlap, flag, k1, k2):
    """
    CUM4EST Fourth-order cumulants.
           Computes sample estimates of fourth-order cumulants
           via the overlapped segment method.
    
           y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)
           y: input data vector (column)
           maxlag: maximum lag
           samp_seg: samples per segment
           overlap: percentage overlap of segments
          flag : 'biased', biased estimates are computed
               : 'unbiased', unbiased estimates are computed.
          k1,k2 : the fixed lags in C3(m,k1) or C4(m,k1,k2); see below
          y_cum : estimated fourth-order cumulant slice
                 C4(m,k1,k2)  -maxlag <= m <= maxlag
    """
    minlag = -maxlag
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecord = (len(signal)-overlap)/(nsamp-overlap)

    nlags = 2 * maxlag +1
    tmp = np.zeros(nlags, dtype=float)
    if flag == "biased":
        scale = np.ones(nlags, dtype=float)/nsamp
    elif flag == "unbiased":
        ind = np.array(range(-maxlag,maxlag+1))
        kmin = min(0, min(k1, k2))
        kmax = max(0, max(k1, k2))
        scale = nsamp - np.array([max(k, kmax) for k in ind]) + np.array([min(k, kmin) for k in ind])
        scale = np.ones(len(scale), dtype=float)/scale
    else:
        raise Exception("The flag should be either 'biased' or 'unbiased'!!")

    mlag = maxlag + max(abs(np.array([k1, k2])))
    mlag = max (mlag, abs(k1-k2))
    nlag = maxlag
    m2k2 = np.zeros(2*maxlag+1, dtype=float)

    if any(signal.imag) != 0:
        complex_flag = 1
    else:
        complex_flag = 0

    y_cum = np.zeros(2*maxlag+1, dtype=float)
    R_yy = np.zeros(2*mlag+1, dtype=float)

    ind = 0
    for i in range(nrecord):
        tmp = y_cum * 0
        x = signal[ind:(ind+nsamp)]
        x = x-float(sum(x))/len(x)
        cx = np.conjugate(x)
        z = x*0

        if k1 >= 0:
            z[:nsamp-k1] = x[:nsamp-k1]*cx[k1:nsamp]
        else:
            z[-k1:nsamp] = x[-k1:nsamp]*cx[:nsamp+k1]
        if k2 >= 0:
            z[:nsamp-k2] = z[:nsamp-k2] * x[k2:nsamp]
            if len(z.shape) == 1:
                z[nsamp-k2:nsamp] = np.zeros(k2)
            else:
                z[nsamp-k2:nsamp] = np.zeros((k2, z.shape[1]))
        else:
            z[-k2:nsamp] = z[-k2:nsamp] * x[:nsamp+k2]
            z[:-k2] = np.zeros[-k2]
        
        tmp[maxlag] = tmp[maxlag] + reduce(lambda m,n:m+n, z*x, 0)
        for k in range(1, maxlag+1):
            tmp[maxlag-k] = tmp[maxlag-k] + reduce(lambda m,n:m+n, z[k:nsamp]*x[:nsamp-k], 0)
            tmp[maxlag+k] = tmp[maxlag+k] + reduce(lambda m,n:m+n, z[:nsamp-k]*x[k:nsamp], 0)
        
        y_cum = y_cum + tmp*scale
        R_yy = cum2est(x,mlag,nsamp,overlap,flag)
        if complex_flag:
            M_yy = cum2x(np.conjugate(x), x, mlag, nsamp, overlap, flag)
        else:
            M_yy = R_yy

        y_cum = y_cum - R_yy[mlag+k1]*R_yy[mlag-k2-nlag:mlag-k2+nlag+1] \
                - R_yy[k1-k2+mlag]*R_yy[mlag-nlag:mlag+nlag+1] \
                - M_yy[mlag+k2]*M_yy[mlag-k1-nlag:mlag-k1+nlag+1]
        ind += nadvance

    return y_cum/nrecord

def test ():
    import scipy.io as sio
    y1 = sio.loadmat("matfile/demo/ma1.mat")['y']
    y = np.load("data/exp_deviate_one.npy")
    # for tesating purpose
    # The right results are:
    #           "biased": [-0.12250513  0.35963613  1.00586945  0.35963613 -0.12250513]
    #           "unbiaed": [-0.12444965  0.36246791  1.00586945  0.36246791 -0.12444965]
    print cum2est(y, 2, 128, 0, 'unbiased')
    
    # For testing 2nd order covariance cummulant
    # biased:   [-0.25719315 -0.12011232  0.35908314  1.01377882  0.35908314 -0.12011232 -0.25719315]
    # unbiased: [-0.26514758 -0.12256359  0.36271024  1.01377882  0.36271024 -0.12256359, -0.26514758]
    print cum2x(sampling(y, 3), sampling(y, 4), 2, 128, 0, "unbiased")

    # For the 3rd cumulant:
    #           "biased": [-0.18203039  0.07751503  0.67113035  0.729953    0.07751503]
    #           "unbiased": [-0.18639911  0.07874543  0.67641484  0.74153955  0.07937539]
    #print cum3est(y, 2, 128, 0, 'unbiased', 1)

    # For testing the 4th-order cumulant
    # "biased": [-0.03642083  0.4755026   0.6352588   1.38975232  0.83791117  0.41641134 -0.97386322]
    # "unbiased": [-0.04011388  0.48736793  0.64948927  1.40734633  0.8445089   0.42303979 -0.99724968]
    #print cum4est(y, 3, 128, 0, 'unbiased', 1, 1)


def cumest (y,norder=2,maxlag=0,nsamp=0,overlap=0,flag='biased',k1=0,k2=0):
    """
    CUMEST Second-, third- or fourth-order cumulants.
         y - time-series  - should be a vector
         norder - cumulant order: 2, 3 or 4 [default = 2]
         maxlag - maximum cumulant lag to compute [default = 0]
         samp_seg - samples per segment  [default = data_length]
         overlap - percentage overlap of segments [default = 0]
                   overlap is clipped to the allowed range of [0,99].
         flag  - 'biased' or 'unbiased'  [default = 'biased']
         k1,k2  - specify the slice of 3rd or 4th order cumulants
         y_cum  - C2(m) or C3(m,k1) or C4(m,k1,k2),  -maxlag <= m <= maxlag
                  depending upon the cumulant order selected
    """
    assert maxlag>0, "maxlag must be non-negative!"
    assert nsamp>=0 and nsamp<len(y), "The number of samples is illigal!"
    if nsamp == 0: nsamp = len(y)

    if norder == 2:
        return cum2est(y, maxlag, nsamp, overlap, flag)
    elif norder == 3:
        return cum3est (y, maxlag, nsamp, overlap, flag, k1)
    elif norder == 4:
        return cum4est (y, maxlag, nsamp, overlap, flag, k1, k2)
    else:
        raise Exception("Cumulant order must be 2, 3, or 4!")

if __name__=="__main__":
    test()


