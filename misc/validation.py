import cumxst as cx
import nested_cumxst as ncx
import numpy as np
import nested_maest as nma
import maest as ma
import impulse_response as ir

winsize = 512
taps = [1, -2.333, 0.667]

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
nl = [4,3,4]
print "With nested sampling:", ncx.cumx(receive, nl, len(nl), len(taps)-1, winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [2,3,5]
print "Without downsampling:", cx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [1,1,1]
print "With PCS downsampling:", cx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)

#######################

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [4,3,4]
print "With nested sampling:", nma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [2,3,5]
print "Without downsampling:", ma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)
pcs = [1,1,1]
print "With PCS downsampling:", ma.maestx (receive, pcs, len(taps)-1, len(pcs), winsize)
