
# for the convinence of simulation
import numpy as np
import impulse_response as ir
import maest as ma

x = np.load("../data/exp_deviate_one_0.npy")
y = ir.moving_average(2, [1, -2.333, 0.667], x)

