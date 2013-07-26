import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import sys, os
import numpy as np


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y/0.04))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'


def hist_plot(m, n, p):
    x = np.arange(0,7)
    #w1 = np.array([m[k/2,1] if k%2==0 else m[k%2,2] for k in range(10)])
    #w2 = np.array([n[k/2,1] if k%2==0 else n[k%2,2] for k in range(10)])
    common_params = dict(bins=10, range=(0,10), normed=False)

    #plt.hist((x,x), weights=(w1,w2), label=["PCS(3)", "PCS(4)"], **common_params)
    #plt.legend(loc=0)
    plt.hist((x,x,x), weights=(m,n,p), **common_params)

    #ax2 = plt.twinx()
    #ax2.plot(x, w1/w2)
    #formatter = FuncFormatter(to_percent)
    #ax2.yaxis.set_major_formatter(formatter)

    #plt.xlabel(r"Length of signal $(10^4)$")
    #plt.ylabel("Proportion of standard deviation and expectation")
    #plt.savefig("convergence_ma3_short_cumulant_pcs123.pdf", fmt='pdf')
    plt.show()

if __name__ == "__main__":
    m3_long = [0.86089537,0.32352935,1.18092441,0.52197473,0.21109557]
    b10000_long = [1.48848624, 0.67883205, 1.90313507, 0.9890348, 0.35486214]
    bchmk_long = [0.66954667, 0.3340954, 0.66633891, 0.30589311, 0.1339124]
    
    m3_short = [0.35177937, 0.02182355]
    b10000_short = [1.82210635,  0.1862351]
    bchmk_short = [0.28251416, 0.01283398]
    
    hist_plot(m3_short+m3_long, b10000_short+b10000_long, bchmk_short+bchmk_long)


