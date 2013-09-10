import numpy as np
from cumxst import cumx
from cumest import cumest
import impulse_response as ir
from multiprocessing import Process
from multiprocessing import Pool
from math import sqrt
from scipy.signal import lfilter

# now it can only deal with one choice of sig_tap
def signal_through_ma_channel(sig_tap, noise_tap, i):
    signal = np.load("/home/creasy/workplace/data/exp_deviate_one_%d.npy"%(i))[:210000]
    receive = ir.moving_average(sig_tap, signal)
    # save signal snr=+inf
    np.save("temp/ma_data_%d.npy"%(i), receive)

    # snr = 0..20 (scale from 1 to 100)
    for j in range(-10,21):
        amp = 10**(j/10.)
        white = np.random.normal(0, sqrt(sum(sig_tap)**2/amp), len(signal))
        color_scale = sqrt(sum(sig_tap)**2/amp/(sum(noise_tap)**2))
        color = ir.moving_average(noise_tap, np.random.normal(0, color_scale, len(signal)))
        np.save("temp/ma_data_white_%d_%d.npy"%(j, i), white+receive)
        np.save("temp/ma_data_color_%d_%d.npy"%(j, i), color+receive)


def ma_gen():
    job = Pool(8)
    r = 50
    ma_sig_tap = [1, 0.9, 0.385, -0.771]
    ma_noise_tap = [1, -2.33, 0.75, 0.5, 0.3, -1.41]

    for i in range(r):
        job.apply_async(signal_through_ma_channel, args=(ma_sig_tap, ma_noise_tap, i))
        print "The %d Monte Carlo data prepared"%(i)

    job.close()
    job.join()


def signal_through_ar_channel(sig_tap, noise_tap, i):
    signal = np.load("/home/creasy/workplace/data/exp_deviate_one_%d.npy"%(i))[:210000]
    receive = lfilter([1], sig_tap, signal)
    # save signal snr=+inf
    np.save("temp/ar_data_%d.npy"%(i), receive)

    # snr = 0..20 (scale from 1 to 100)
    for j in range(-10,21):
        amp = 10**(j/10.)
        white = np.random.normal(0, sqrt(sum(sig_tap)**2/amp), len(signal))
        color_scale = sqrt(sum(sig_tap)**2/amp/(sum(noise_tap)**2))
        color = lfilter([1], noise_tap, np.random.normal(0, color_scale, len(signal)))
        np.save("temp/ar_data_white_%d_%d.npy"%(j, i), white+receive)
        np.save("temp/ar_data_color_%d_%d.npy"%(j, i), color+receive)

def ar_gen():
    job = Pool(8)
    r = 50
    ar_sig_tap = [1, -1.4, 0.65]
    ar_noise_tap = [1, 0.3, 0.745]

    for i in range(r):
        job.apply_async(signal_through_ar_channel, args=(ar_sig_tap, ar_noise_tap, i))
        print "The %d Monte Carlo data prepared"%(i)

    job.close()
    job.join()


if __name__ == "__main__":
    ar_gen()


