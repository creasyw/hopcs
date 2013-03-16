# Used in the benchmark for signal processing without coprime sampling
def autocorr(x):
    """Return result of autocorrelation of sequence without any sampling"""
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

# Used in the benchmark for signal processing without coprime sampling
def fourier (r_xx, Fs, NFFT, hamming, overlap=True, sides='default'):
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
        temp = r_xx[ind[i]:(ind[i]+NFFT)]*hamming
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

def benchmark(signal, Fs, NFFT):
    """ Perform autocorrelation-->spectrogram tranformation without coprime sampling """
    window_size = NFFT
    length = len(signal)
    steps = int(ceil(length/float(window_size)))
    r_xx = np.zeros(len(signal))
    overlap = False
    for i in range(steps):
        upper = min((i+1)*window_size, length)
        r_xx[i*window_size:upper] = autocorr(signal[i*window_size:upper])
    hamming = np.hamming(NFFT)
    psd, freq, time = fourier(r_xx, Fs, NFFT, hamming, overlap)
    return abs(psd), freq, time

