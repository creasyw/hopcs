import numpy as np

def cum2est (signal, maxlag, nsamp, overlap, flag):
    """
    CUM2EST Covariance function.
       Should be involed via "CUMEST" for proper parameter checks.
       y_cum = cum2est (y, maxlag, samp_seg, overlap,  flag)

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

def test ():
    # for tesating purpose
    # The right results are:
    #           "biased": [-0.12250513  0.35963613  1.00586945  0.35963613 -0.12250513]
    #           "unbiaed": [-0.12444965  0.36246791  1.00586945  0.36246791 -0.12444965]
    import scipy.io as sio
    y = sio.loadmat("matfile/demo/ma1.mat")['y']
    print cum2est(y, 2, 128, 0, 'unbiased')

if __name__=="__main__":


