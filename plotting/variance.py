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

    return percentage(np.array(result))

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'

def plotting5(m):
    xlen = range(len(m))
    plt.plot(xlen, m[:,0], "-.", label=r'$C_{3y}(-2,0)$')
    plt.plot(xlen, m[:,1], "--", label=r'$C_{3y}(-1,0)$')
    plt.plot(xlen, m[:,2], "-", label=r'$C_{3y}(0,0)$')
    plt.plot(xlen, m[:,3], ":", label=r'$C_{3y}(1,0)$')
    plt.plot(xlen, m[:,4], "o", label=r'$C_{3y}(2,0)$')
    
    plt.ylim((0,1))
    plt.xlabel(r"Length of signal $(10^4)$")
    plt.ylabel("Proportion of standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2)
    plt.savefig("convergence_ma3_short_cumulant_pcs123.pdf", fmt='pdf')

def plotting11(m):
    xlen = range(len(m))
    ax = plt.gca()
    plt.plot(xlen, m[:,0], ":", label=r'$C_{3y}(-5,0)$')
    plt.plot(xlen, m[:,1], "--", label=r'$C_{3y}(-4,0)$')
    plt.plot(xlen, m[:,2], "-.", label=r'$C_{3y}(-3,0)$')
    plt.plot(xlen, m[:,3], "o", label=r'$C_{3y}(-2,0)$')
    plt.plot(xlen, m[:,4], "^", label=r'$C_{3y}(-1,0)$')
    plt.plot(xlen, m[:,5], "-", label=r'$C_{3y}(0,0)$')
    plt.plot(xlen, m[:,6], "<", label=r'$C_{3y}(1,0)$')
    plt.plot(xlen, m[:,7], ">", label=r'$C_{3y}(2,0)$')
    plt.plot(xlen, m[:,8], "s", label=r'$C_{3y}(3,0)$')
    plt.plot(xlen, m[:,9], "p", label=r'$C_{3y}(4,0)$')
    plt.plot(xlen, m[:,10], "*", label=r'$C_{3y}(5,0)$')

    plt.ylim((0,1.5))
    plt.xlabel(r"Length of signal $(10^4)$")
    plt.ylabel("Proportion of standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2)
    plt.savefig("convergence_ma3_long_cumulant_pcs123.pdf", fmt='pdf')
    plt.show()

if __name__ == "__main__":
    filename = "convergence_ma3_short_cumulant_pcs123.txt"
    result = calculte(filename)
    plotting5(result)


