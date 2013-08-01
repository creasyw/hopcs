import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import sys, os
import numpy as np


def to_percent(y, position):
    s = str(int(100 * y))
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'


def hist_plot(m, n, p):
    x = np.arange(0,7)
    ax = plt.subplot(111)
    common_params = dict(bins=7, range=(0,7), normed=False)

    plt.hist((x,x,x), weights=(m,n,p), **common_params)

    ax2 = plt.twinx()
    ax2.plot(map(lambda i:i+0.5, x), np.array(m)/np.array(n), 'k-.', linewidth=2)
    formatter = FuncFormatter(to_percent)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylabel("Ratio of RMS between PCS-HOS and HOS", labelpad=10)

    names = ["MA(2)_tap1","MA(2)_tap2","MA(5)_tap1","MA(5)_tap2","MA(5)_tap3","MA(5)_tap4","MA(5)_tap5"]
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    ax.set_xticks([k+0.5 for k in range(7)])
    ax.set_xticklabels(names,rotation=30, rotation_mode="anchor", ha="right")
    ax.set_ylabel("Root-mean-square error", labelpad=10)
    ax.set_xlabel("Taps from MA(2) or MA(5) system", labelpad=10)
    plt.tight_layout()

    plt.savefig("pcs3_vs_benchmark.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    m3_long = [0.86089537,0.32352935,1.18092441,0.52197473,0.21109557]
    b10000_long = [1.48848624, 0.67883205, 1.90313507, 0.9890348, 0.35486214]
    bchmk_long = [0.66954667, 0.3340954, 0.66633891, 0.30589311, 0.1339124]
    
    m3_short = [0.35177937, 0.02182355]
    b10000_short = [1.82210635,  0.1862351]
    bchmk_short = [0.28251416, 0.01283398]
    
    hist_plot(m3_short+m3_long, b10000_short+b10000_long, bchmk_short+bchmk_long)


