import nested_cumxst as cx
import nested_cumxst as ncx
import numpy as np
import impulse_response as ir

signal = np.load("../data/exp_deviate_one_0.npy")[:10000]
receive = ir.moving_average(taps, signal)

win = 512
nl = [4,3,4]
t1 = ncx.cum2x(receive, nl, len(taps)-1, winsize, 0)

pcs = [2,3]
t2 = = cx.cumx(receive, pcs, len(pcs), len(taps)-1, winsize)
