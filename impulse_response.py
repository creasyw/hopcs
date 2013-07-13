import numpy as np

def moving_average(vt, input):
    """
    Simulate the MA system with the # of lag, the impulse response of each
    tap is stored in the list vt. Input is signal with format of 1D numpy.ndarray.
    """
    taps = len(vt)
    temp = np.hstack((np.zeros(taps), input))
    return np.array([sum(temp[k:k-taps:-1]*vt) for k in range(taps, len(temp))])



