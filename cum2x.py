import numpy as np

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


