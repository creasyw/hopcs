import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import sys, os
import numpy as np

def percentage (m):
    # standard deviation / original values
    return np.sqrt(m[1::2,:])/ np.absolute(m[0::2,:])


def calculte(filename):
    finder = re.compile("\[|-?\d+\.\d*[eE]?[-+]?\d*|\]")
    result = []
    temp = []
    with open(os.path.join(os.path.dirname(__file__), filename)) as datafile:
        for row in datafile:
            hit = finder.findall(row)
            if "[" in hit:
                hit.remove("[")
                temp = hit
                if "]" in hit:
                    temp.remove("]")
                    result.append(map(lambda x: float(x), temp))
            elif "]" in hit:
                hit.remove("]")
                temp += hit
                result.append(map(lambda x: float(x), temp))
            else:
                temp += hit

    return np.array(result)[2::3,:]

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y/0.04))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'


def hist_plot(m, n):
    x = np.arange(0,10)
    w1 = np.array([m[k/2,1] if k%2==0 else m[k%2,2] for k in range(10)])
    w2 = np.array([n[k/2,1] if k%2==0 else n[k%2,2] for k in range(10)])
    common_params = dict(bins=10, range=(0,10), normed=True)

    plt.hist((x,x), weights=(w1,w2), label=["PCS(3)", "PCS(4)"], **common_params)
    plt.legend(loc=0)

    ax2 = plt.twinx()
    ax2.plot(x, w1/w2)
    formatter = FuncFormatter(to_percent)
    ax2.yaxis.set_major_formatter(formatter)

    #plt.xlabel(r"Length of signal $(10^4)$")
    #plt.ylabel("Proportion of standard deviation and expectation")
    #plt.savefig("convergence_ma3_short_cumulant_pcs123.pdf", fmt='pdf')
    plt.show()

if __name__ == "__main__":
    hos3 = calculte("rms_winsize_short_pcs123.txt")
    hos4 = calculte("rms_winsize_short_pcs1123.txt")
    hist_plot(hos3, hos4)


