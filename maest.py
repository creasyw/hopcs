import numpy as np
from scipy.linalg import toeplitz, lstsq
from cumest import cumest
from pcs_cumest import cumx

def maestx(y, pcs, q, norder=3,samp_seg=1,overlap=0):
    """
    MAEST  MA parameter estimation via the GM-RCLS algorithm, with Tugnait's fix
        y  - time-series (vector or matrix)
        q  - MA order
        norder - cumulant-order to use  [default = 3]
        samp_seg - samples per segment for cumulant estimation
                  [default: length of y]
        overlap - percentage overlap of segments  [default = 0]
        flag - 'biased' or 'unbiased'          [default = 'biased']
        Return: estimated MA parameter vector
    """
    assert norder>=2 and norder<=4, "Cumulant order must be 2, 3, or 4!"
    nsamp = len(y)
    overlap = max(0, min(overlap,99))

    c2 = cumx(y, pcs, 2,q, samp_seg, overlap)
    c2 = np.hstack((c2, np.zeros(q)))
    cumd = cumx(y, pcs, norder,q,samp_seg,overlap,0,0)[::-1]
    cumq = cumx(y, pcs, norder,q,samp_seg,overlap,q,0)
    cumd = np.hstack((cumd, np.zeros(q)))
    cumq[:q] = np.zeros(q)

    cmat = toeplitz(cumd, np.hstack((cumd[0],np.zeros(q))))
    rmat = toeplitz(c2,   np.hstack((c2[0],np.zeros(q))))
    amat0 = np.hstack((cmat, -rmat[:,1:q+1]))
    rvec0 = c2

    cumq = np.hstack((cumq[2*q:q-1:-1], np.zeros(q)))
    cmat4 = toeplitz(cumq, np.hstack((cumq[0],np.zeros(q))))
    c3   = cumd[:2*q+1]
    amat0 = np.vstack((np.hstack((amat0, np.zeros((3*q+1,1)))), \
            np.hstack((np.hstack((np.zeros((2*q+1,q+1)), cmat4[:,1:q+1])), \
            np.reshape(-c3,(len(c3),1))))))
    rvec0 = np.hstack((rvec0, -cmat4[:,0]))

    row_sel = range(q)+range(2*q+1,3*q+1)+range(3*q+1,4*q+1)+range(4*q+2,5*q+2)
    amat0 = amat0[row_sel,:]
    rvec0 = rvec0[row_sel]

    bvec = lstsq(amat0, rvec0)[0]
    b1 = bvec[1:q+1]/bvec[0]
    b2 = bvec[q+1:2*q+1]
    if norder == 3:
        if all(b2 > 0):
            b1 = np.sign(b1) * np.sqrt(0.5*(b1**2 + b2))
        else:
            print 'MAEST: alternative solution b1 used'
    else:
        if all(np.sign(b2) == np.sign(b1)):
            b1 = np.sign(b1)* (abs(b1) + abs(b2)**(1./3))/2
        else:
            print 'MAEST: alternative solution b1 used'
    return np.hstack(([1], b1))


def maest(y,q, norder=3,samp_seg=1,overlap=0,flag='biased'):
    """
    MAEST  MA parameter estimation via the GM-RCLS algorithm, with Tugnait's fix
        y  - time-series (vector or matrix)
        q  - MA order
        norder - cumulant-order to use  [default = 3]
        samp_seg - samples per segment for cumulant estimation
                  [default: length of y]
        overlap - percentage overlap of segments  [default = 0]
        flag - 'biased' or 'unbiased'          [default = 'biased']
        Return: estimated MA parameter vector
    """
    assert norder>=2 and norder<=4, "Cumulant order must be 2, 3, or 4!"
    nsamp = len(y)
    overlap = max(0, min(overlap,99))

    c2 = cumest(y,2,q, samp_seg, overlap, flag)
    c2 = np.hstack((c2, np.zeros(q)))
    cumd = cumest(y,norder,q,samp_seg,overlap,flag,0,0)[::-1]
    cumq = cumest(y,norder,q,samp_seg,overlap,flag,q,0)
    cumd = np.hstack((cumd, np.zeros(q)))
    cumq[:q] = np.zeros(q)

    cmat = toeplitz(cumd, np.hstack((cumd[0],np.zeros(q))))
    rmat = toeplitz(c2,   np.hstack((c2[0],np.zeros(q))))
    amat0 = np.hstack((cmat, -rmat[:,1:q+1]))
    rvec0 = c2

    cumq = np.hstack((cumq[2*q:q-1:-1], np.zeros(q)))
    cmat4 = toeplitz(cumq, np.hstack((cumq[0],np.zeros(q))))
    c3   = cumd[:2*q+1]
    amat0 = np.vstack((np.hstack((amat0, np.zeros((3*q+1,1)))), \
            np.hstack((np.hstack((np.zeros((2*q+1,q+1)), cmat4[:,1:q+1])), \
            np.reshape(-c3,(len(c3),1))))))
    rvec0 = np.hstack((rvec0, -cmat4[:,0]))

    row_sel = range(q)+range(2*q+1,3*q+1)+range(3*q+1,4*q+1)+range(4*q+2,5*q+2)
    amat0 = amat0[row_sel,:]
    rvec0 = rvec0[row_sel]

    bvec = lstsq(amat0, rvec0)[0]
    b1 = bvec[1:q+1]/bvec[0]
    b2 = bvec[q+1:2*q+1]
    if norder == 3:
        if all(b2 > 0):
            b1 = np.sign(b1) * np.sqrt(0.5*(b1**2 + b2))
        else:
            print 'MAEST: alternative solution b1 used'
    else:
        if all(np.sign(b2) == np.sign(b1)):
            b1 = np.sign(b1)* (abs(b1) + abs(b2)**(1./3))/2
        else:
            print 'MAEST: alternative solution b1 used'
    return np.hstack(([1], b1))


def test():
    import scipy.io as sio
    y = sio.loadmat("matfile/demo/ma1.mat")['y']
    # [ 1.          0.97135456  0.38142558 -0.77589961]
    print maest(y, 3, 3, 128)
    # [ 1.          0.9607745   0.44815601 -0.73434479]
    print maest(y, 3, 4, 256)


if __name__=="__main__":
    test()




