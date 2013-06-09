import numpy as np
from cumest import cum2est, cum3est, cum4est

def sampling (signal, factor):
    """
    Return signals with sampling period given with "factor", and stuff zeros in the interval.
    The format of return values is np.ndarray.
    NOTE: it is different from the same function in "sampling.py".
    """
    return np.array([signal[k] if k%factor==0 else 0 for k in range(len(signal))])

def cum2x (x,y, maxlag, nsamp, overlap):
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
        xs = np.array([j-float(sum(xs))/sum(1 for i in xs if i!=0) if j!=0 else 0 for j in xs])
        ys = y[ind:(ind+nsamp)]
        ys = np.array([j-float(sum(ys))/sum(1 for i in ys if i!=0) if j!=0 else 0 for j in ys])
        temp = xs*ys
        y_cum[maxlag] += reduce(lambda m,n:m+n,temp, 0)
        count[maxlag] += sum(1 for i in temp if i!=0)
        for m in range(1,maxlag+1):
            temp = xs[m:nsamp]*ys[:nsamp-m]
            y_cum[maxlag-m] = y_cum[maxlag-m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag-m] += sum(1 for i in temp if i!=0)
            temp = xs[:nsamp-m]*ys[m:nsamp]
            y_cum[maxlag+m] = y_cum[maxlag+m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag+m] += sum(1 for i in temp if i!=0)
        ind += nadvance
#    if flag == "biased":
#        scale = np.ones(nlags, dtype=float)/nsamp/nrecs
#    elif flag == "unbiased":
#        scale = np.array(range(nsamp-maxlag,nsamp+1)+range(nsamp-1,nsamp-maxlag-1,-1))
#        scale = np.ones(2*maxlag+1, dtype=float)/scale
#    else:
#        raise Exception("The flag should be either 'biased' or 'unbiased'!!")
    scale = 1./count
    return y_cum*scale


def cum3x (x, y, z, maxlag=0, nsamp=1, overlap=0, k1=0):
    """
    CUM3X Third-order cross-cumulants.
        x,y,z  - data vectors/matrices with identical dimensions
                 if x,y,z are matrices, rather than vectors, columns are
                 assumed to correspond to independent realizations,
                 overlap is set to 0, and samp_seg to the row dimension.
        maxlag - maximum lag to be computed    [default = 0]
        samp_seg - samples per segment  [default = data_length]
        overlap - percentage overlap of segments [default = 0]
                  overlap is clipped to the allowed range of [0,99].
        flag : 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.
        k1: the fixed lag in c3(m,k1): defaults to 0
    Return:
        y_cum:  estimated third-order cross cumulant,
                E x^*(n)y(n+m)z(n+k1),   -maxlag <= m <= maxlag
    """
    assert len(x) == len(y) == len(z), "the length of signal should be the same!"
    assert 0<=nsamp<=len(x), "The length of segment is illegal."
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecs  = (len(x)-overlap)/nadvance
    nlags = 2*maxlag+1
    y_cum = np.zeros(nlags, dtype=float)
    count = np.zeros(nlags, dtype=float)

    if k1 >= 0:
        indx = range(nsamp-k1)
        indz = range(k1, nsamp)
    else:
        indx = range(-k1, nsamp)
        indz = range(nsamp+k1)
    ind = 0
    for k in range(nrecs):
        xs = x[ind:(ind+nsamp)]
        xs = np.array([j-float(sum(xs))/sum(1 for i in xs if i!=0) if j!=0 else 0 for j in xs])
        ys = y[ind:(ind+nsamp)]
        ys = np.array([j-float(sum(ys))/sum(1 for i in ys if i!=0) if j!=0 else 0 for j in ys])
        zs = z[ind:(ind+nsamp)]
        zs = np.conjugate(np.array([j-float(sum(zs))/sum(1 for i in zs if i!=0) if j!=0 else 0 for j in zs]))

        u = np.zeros(nsamp, dtype=float)
        u[indx[0]:indx[-1]+1] = xs[indx]*zs[indz]

        temp = u*ys
        y_cum[maxlag] += reduce(lambda m,n:m+n,temp, 0)
        count[maxlag] += sum(1 for i in temp if i!=0)
        for m in range(1,maxlag+1):
            temp = u[m:nsamp]*ys[:nsamp-m]
            y_cum[maxlag-m] = y_cum[maxlag-m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag-m] += sum(1 for i in temp if i!=0)
            temp = u[:nsamp-m]*ys[m:nsamp]
            y_cum[maxlag+m] = y_cum[maxlag+m]+reduce(lambda i,j:i+j,temp, 0)
            count[maxlag+m] += sum(1 for i in temp if i!=0)
        ind += nadvance
#    if flag == "biased":
#        scale = np.ones(nlags, dtype=float)/nsamp/nrecs
#    elif flag == "unbiased":
#        lsamp = nsamp-abs(k1)
#        scale = np.array(range(lsamp-maxlag,lsamp+1)+range(lsamp-1,lsamp-maxlag-1,-1))
#        scale = np.ones(len(scale), dtype=float)/scale/nrecs
#    else:
#        raise Exception("The flag should be either 'biased' or 'unbiased'!!")
    scale = 1./count
    return y_cum*scale


#def cum4x (w, x, y, z, maxlag=0, nsamp=1, overlap=0, flag='unbiased', k1=0, k2=0):
def cum4x (w, x, y, z, maxlag=0, nsamp=1, overlap=0, k1=0, k2=0):
    """
    CUM4EST Fourth-order cumulants.
           Computes sample estimates of fourth-order cumulants
           via the overlapped segment method.
    
           y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)
           y: input data vector (column)
           maxlag: maximum lag
           samp_seg: samples per segment
           overlap: percentage overlap of segments
           flag : 'biased', biased estimates are computed     (DISABLED)
                  'unbiased', unbiased estimates are computed.
           k1,k2 : the fixed lags in C3(m,k1) or C4(m,k1,k2); see below
           y_cum : estimated fourth-order cumulant slice
                  C4(m,k1,k2)  -maxlag <= m <= maxlag
    """
    length = len(x)
    assert length==len(y)==len(z)==len(w), "The four input signals should have same length!"
    assert maxlag>=0, "maxlag should be nonnegative!"
    assert 0<nsamp<=length, "The segmentation setting is illegal!"

    overlap0 = overlap
    overlap = overlap/100*nsamp
    nadvance = nsamp - overlap
    nrecs = (length-overlap)/nadvance
    nlags = 2 * maxlag +1

#    if flag == "biased":
#        scale = np.ones(nlags, dtype=float)/nsamp
#        sc1 = sc2 = sc12 = 1./nsamp
#    elif flag == "unbiased":
#        ind = np.array(range(-maxlag,maxlag+1))
#        kmin = min(0, min(k1, k2))
#        kmax = max(0, max(k1, k2))
#        scale = nsamp - np.array([max(k, kmax) for k in ind]) + np.array([min(k, kmin) for k in ind])
#        scale = np.ones(nlags, dtype=float)/scale
#        sc1 = 1./(nsamp-abs(k1))
#        sc2 = 1./(nsamp-abs(k2))
#        sc12 = 1./(nsamp-abs(k1-k2))
#    else:
#        raise Exception("The flag should be either 'biased' or 'unbiased'!!")

    y_cum = np.zeros(nlags, dtype=float)
    rind = -np.array(range(-maxlag, maxlag+1))
    ind = 0
    for i in range(nrecs):
        # different from 2nd- and 3rd- order
        # only consider the non-zero elements within every segment
        count = np.zeros(nlags, dtype=float)
        tmp = y_cum * 0
        R_zy = R_wy = M_wz = 0

        ws = w[ind:(ind+nsamp)]
        ws = np.array([j-float(sum(ws))/sum(1 for i in ws if i!=0) if j!=0 else 0 for j in ws])
        xs = x[ind:(ind+nsamp)]
        xs = np.array([j-float(sum(xs))/sum(1 for i in xs if i!=0) if j!=0 else 0 for j in xs])
        zs = z[ind:(ind+nsamp)]
        zs = np.array([j-float(sum(zs))/sum(1 for i in zs if i!=0) if j!=0 else 0 for j in zs])
        ys = y[ind:(ind+nsamp)]
        ys = np.array([j-float(sum(ys))/sum(1 for i in ys if i!=0) if j!=0 else 0 for j in ys])
        cys = np.conjugate(ys)
        ziv = xs*0

        if k1 >= 0:
            ziv[:nsamp-k1] = ws[:nsamp-k1]*cys[k1:nsamp]
            temp = ws[:nsamp-k1]*ys[k1:nsamp]
            R_wy += reduce(lambda m, n: m+n, temp, 0)
            sc1 = sum(1 for m in temp if m!=0)
        else:
            ziv[-k1:nsamp] = ws[-k1:nsamp]*cys[:nsamp+k1]
            temp = ws[-k1:nsamp]*ys[:nsamp+k1]
            R_wy += reduce(lambda m, n: m+n, temp, 0)
            sc1 = sum(1 for m in temp if m!=0)
        if k2 >= 0:
            ziv[:nsamp-k2] = ziv[:nsamp-k2] * zs[k2:nsamp]
            if len(z.shape) == 1:
                ziv[nsamp-k2:nsamp] = np.zeros(k2)
            else:
                ziv[nsamp-k2:nsamp] = np.zeros((k2, z.shape[1]))
            temp = ws[:nsamp-k2]*zs[k2:nsamp]
            M_wz += reduce(lambda m, n: m+n, temp, 0)
            sc2 = sum(1 for m in temp if m!=0)
        else:
            ziv[-k2:nsamp] = ziv[-k2:nsamp] * zs[:nsamp+k2]
            ziv[:-k2] = np.zeros[-k2]
            temp = ws[-k2:nsamp]*zs[:nsamp-k2]
            M_wz += reduce(lambda m, n: m+n, temp, 0)
            sc2 = sum(1 for m in temp if m!=0)
        if k1-k2 >= 0:
            temp = zs[:nsamp-k1+k2]*ys[k1-k2:nsamp]
            R_zy += reduce(lambda m, n: m+n, temp, 0)
            sc12 = sum(1 for m in temp if m!=0)
        else:
            temp = zs[-k1+k2:nsamp]*ys[:nsamp-k2+k1]
            R_zy += reduce(lambda m, n: m+n, temp, 0)
            sc12 = sum(1 for m in temp if m!=0)
        
        temp = ziv*xs
        tmp[maxlag] += reduce(lambda m,n:m+n, temp, 0)
        count[maxlag] += sum(1 for m in temp if m!=0)
        for k in range(1, maxlag+1):
            temp = ziv[k:nsamp]*xs[:nsamp-k]
            tmp[maxlag-k] += reduce(lambda m,n:m+n, temp, 0)
            count[maxlag-k] += sum(1 for m in temp if m!=0)
            temp = ziv[:nsamp-k]*xs[k:nsamp]
            tmp[maxlag+k] += reduce(lambda m,n:m+n, temp, 0)
            count[maxlag+k] += sum(1 for m in temp if m!=0)
        
        y_cum += [tmp[m]/count[m] if count[m]!=0 else 0 for m in range(len(count))]
        R_wx = cum2x(ws, xs, maxlag, nsamp, overlap0)
        R_zx = cum2x(zs, xs, maxlag+abs(k2), nsamp, overlap0)
        M_yx = cum2x(cys, xs, maxlag+abs(k1), nsamp, overlap0)

        y_cum = y_cum - R_zy*R_wx/sc12 - R_wy*R_zx[-k2+abs(k2):2*maxlag-k2+abs(k2)+1] /sc1 \
                - M_wz*M_yx[-k1+abs(k1):2*maxlag-k1+abs(k1)+1]/sc2
        ind += nadvance

    return y_cum/nrecs


def test ():
    import scipy.io as sio
    y = sio.loadmat("matfile/demo/ma1.mat")['y']
    y = y.flatten()
    #y = np.load("data/exp_deviate_one.npy")
    
    # The 'biased'/'unbiased' flag is disabled for the application of PCS
    # The output result are unbiased estimate of cumulant
    
    # For testing 2nd order covariance cummulant.
    # unbiased: [-0.26338305 -0.12444965  0.36246791  1.00586945  0.36246791 -0.12444965 -0.26338305]
    print cum2est(y, 3, 128, 0, 'unbiased')
    print cum2x(y, y, 3, 128, 0)
    print cum2x(sampling(y,2), sampling(y,3), 3, 128, 0)

    # For the 3rd covariance cumulant: cum3x(y, y, y, 2, 128, 0, 'unbiased', 0)
    # biased: [ 0.43001919  0.729953    0.75962972  0.67113035 -0.15817154]
    # unbiased: [ 0.43684489  0.73570066  0.75962972  0.67641484 -0.1606822 ]
    #print cum3est(y, 2, 128, 0, 'unbiased', 0)
    print "\n\n"
    print cum3est(y, 2, 128, 0, 'unbiased', 2)
    print cum3x(y, y, y, 2, 128, 0, 2)
    print cum3x(sampling(y,2), sampling(y,3), sampling(y,5), 2, 512, 0, 2)

    # For testing the 4th-order cumulant
    # "biased": [-0.55006876  0.83791117  2.66759034  1.65635663 -0.32747939]
    # "unbiased": [-0.55880001  0.8445089   2.66759034  1.66939881 -0.33267748]
    print "\n\n"
    print cum4est(y, 2, 512, 0, 'unbiased', 0, 0)
    print cum4x(y, y, y, y, 2, 512, 0, 0, 0)
    # NOTE: the length of segmentation should larger than the product of PCS factors
    print cum4x(sampling(y,2), sampling(y,3), sampling(y,5), sampling(y,7), 2, 512, 0, 0, 0)


def cumx (y, pcs, norder=2,maxlag=0,nsamp=0,overlap=0,k1=0,k2=0):
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
        assert len(pcs)>=2, "There is not sufficient PCS coefficients!"
        return cum2x (sampling(y,pcs[0]), sampling(y,pcs[1]), maxlag, nsamp, overlap)
    elif norder == 3:
        assert len(pcs)>=3, "There is not sufficient PCS coefficients!"
        return cum3x (sampling(y,pcs[0]), sampling(y,pcs[1]), sampling(y,pcs[2]), \
                maxlag, nsamp, overlap, k1)
    elif norder == 4:
        assert len(pcs)>=4, "There is not sufficient PCS coefficients!"
        return cum4x (sampling(y,pcs[0]), sampling(y,pcs[1]), sampling(y,pcs[2]), \
                sampling(y,pcs[3]), maxlag, nsamp, overlap, k1, k2)
    else:
        raise Exception("Cumulant order must be 2, 3, or 4!")

if __name__=="__main__":
    test()


