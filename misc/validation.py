import nested_cumxst as cx
import nested_cumxst as ncx
import numpy as np
import impulse_response as ir

winsize = 512
taps = [1, -2.333, 0.667]

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
nl = [4,3,4]
print "With nested sampling:", ncx.cumx(signal, nl, len(nl), len(taps)-1, winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [2,3,5]
print "Without downsampling:", cx.cumx(signal, pcs, len(pcs), len(taps)-1, winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [1,1,1]
print "With PCS downsampling:", cx.cumx(signal, pcs, len(pcs), len(taps)-1, winsize)
