import numpy as np

def moving_average(lag, vt, input):
    """
    Simulate the MA system with the # of lag, the impulse response of each
    tap is stored in the list vt. Input is signal with format of 1D numpy.ndarray.
    Note that using lag rather than taps in the command line is to be
    consistent with maest and cumest.
    """
    taps = lag+1
    assert len(vt) == taps, "The number of taps is not equal to the length of IR."
    temp = np.hstack((np.zeros(taps), input))
    return np.array([sum(temp[k:k-taps:-1]*vt) for k in range(taps, len(temp))])



