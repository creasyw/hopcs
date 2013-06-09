import numpy as np

def moving_average(nt, vt, input):
    """
    Simulate the MA system with the # of nt taps, the impulse response of each
    tap is stored in the list vt. Input is signal with format of 1D numpy.ndarray.
    """
    assert len(vt) == nt, "The number of taps is not equal to the length of IR."
    temp = np.hstack((np.zeros(nt), input))
    return np.array([sum(temp[k:k-nt:-1]*vt) for k in range(nt, len(temp))])



