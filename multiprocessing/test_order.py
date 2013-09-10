from multiprocessing import Process
from multiprocessing import Pool
from maorder import pcs_cx, hos_cm
from arorder import pcs_ar

def ma_order():
    job = Pool(8)
    r = 50
    winsize = 512

    # snr = +inf
    pcs = [1,2,3]
    order = 6
    for slicing in range(5000,50001,5000):
        job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, 500, "white"))
        for snr in range(-10,21):
            job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, snr, "white"))
            job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, snr, "color"))
#    pcs = [1,1,2,3]
#    for slicing in range(5000,50001,5000):
#        job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, 500, "white"))
#        for snr in range(-10,21):
#            job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, snr, "white"))
#            job.apply_async(pcs_cx, args=(pcs, order, winsize, r, slicing, snr, "color"))

    #benchmark for non-PCS
    slicing = 5000
    job.apply_async(hos_cm, args=(2, order, winsize, r, slicing, 500, "white"))
    job.apply_async(hos_cm, args=(3, order, winsize, r, slicing, 500, "white"))
    for snr in range(-10,21):
        job.apply_async(hos_cm, args=(2, order, winsize, r, slicing, snr, "white"))
        job.apply_async(hos_cm, args=(3, order, winsize, r, slicing, snr, "white"))
        job.apply_async(hos_cm, args=(2, order, winsize, r, slicing, snr, "color"))
        job.apply_async(hos_cm, args=(3, order, winsize, r, slicing, snr, "color"))

    job.close()
    job.join()

def ar_order():
    job = Pool(8)
    r = 50
    winsize = 512

    # snr = +inf
    pcs = [1,2,3]
    ar = 5
    ma = 0
    for slicing in range(5000,50001,10000):
        job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, 500, "white"))
        for snr in range(-10,21):
            job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, snr, "white"))
            job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, snr, "color"))

    #benchmark for non-PCS
    slicing = 5000
    pcs = [1,1,1]
    job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, 500, "white"))
    for snr in range(-10,21):
        job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, snr, "white"))
        job.apply_async(pcs_ar, args=(pcs, ar, ma, winsize, r, slicing, snr, "color"))

    job.close()
    job.join()

if __name__ == "__main__":
    ar_order()


