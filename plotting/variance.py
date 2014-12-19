import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import sys, os
import numpy as np

long_taps = np.array([1, 0.1, -1.87, 3.02, -1.435, 0.49])


def percentage (m):
    return np.sqrt(m[1::2,:])/ np.absolute(m[0::2,:])

def percentage_3row (m):
    return np.sqrt(m[1::3,:])/ np.absolute(m[0::3,:])


def calculate(filename):
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

    #return percentage(np.array(result))
    return np.array(result)

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'

def plottingcx5(m):
    xlen = range(len(m))
    plt.plot(xlen, m[:,0], linestyle=":", marker="v", label=r'$c_{3y}(-2,0)$')
    plt.plot(xlen, m[:,1], linestyle=":", marker="^", label=r'$c_{3y}(-1,0)$')
    plt.plot(xlen, m[:,2], "-", label=r'$c_{3y}(0,0)$')
    plt.plot(xlen, m[:,3], linestyle=":", marker="<", label=r'$c_{3y}(1,0)$')
    plt.plot(xlen, m[:,4], linestyle=":", marker=">", label=r'$c_{3y}(2,0)$')
    
    plt.ylim((0,1))
    plt.xlabel(r"Length of signal $(\times 10^4)$")
    plt.ylabel("Ratio between standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2, fontsize=15)
    plt.grid()
    plt.savefig(filename, format='pdf')
    plt.show()

def plottingcx11(m):
    xlen = range(len(m))
    ax = plt.gca()
    plt.plot(xlen, m[:,0], linestyle=":", marker="*", label=r'$c_{3y}(-5,0)$')
    plt.plot(xlen, m[:,1], linestyle=":", marker="p", label=r'$c_{3y}(-4,0)$')
    plt.plot(xlen, m[:,2], linestyle=":", marker="v", label=r'$c_{3y}(-3,0)$')
    plt.plot(xlen, m[:,3], linestyle=":", marker="^", label=r'$c_{3y}(-2,0)$')
    plt.plot(xlen, m[:,4], linestyle=":", marker="h", label=r'$c_{3y}(-1,0)$')
    plt.plot(xlen, m[:,5], "-", label=r'$c_{3y}(0,0)$')
    plt.plot(xlen, m[:,6], linestyle=":", marker="d", label=r'$c_{3y}(1,0)$')
    plt.plot(xlen, m[:,7], linestyle=":", marker="<", label=r'$c_{3y}(2,0)$')
    plt.plot(xlen, m[:,8], linestyle=":", marker=">", label=r'$c_{3y}(3,0)$')
    plt.plot(xlen, m[:,9], linestyle=":", marker="s", label=r'$c_{3y}(4,0)$')
    plt.plot(xlen, m[:,10], linestyle=":", marker="o", label=r'$c_{3y}(5,0)$')

    plt.ylim((0,1.5))
    plt.xlabel(r"Length of signal $(\times 10^4)$")
    plt.ylabel("Ratio between standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2, fontsize=15)
    plt.grid()
    plt.savefig(filename, format='pdf')
    plt.show()

def plottingma5(m):
    xlen = range(len(m))
    ax = plt.gca()
    plt.plot(xlen, m[:,1], "-", label=r'$tap_1$')
    plt.plot(xlen, m[:,2], linestyle=":", marker='v', label=r'$tap_2$')
    plt.plot(xlen, m[:,3], linestyle=":", marker='^', label=r'$tap_3$')
    plt.plot(xlen, m[:,4], linestyle=":", marker='<', label=r'$tap_4$')
    plt.plot(xlen, m[:,5], linestyle=":", marker='>', label=r'$tap_5$')

    plt.ylim((0,2))
    plt.xlabel(r"Length of signal $(\times 10^4)$")
    plt.ylabel("Ratio between standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2, fontsize=15)
    plt.grid()
    plt.savefig(filename, format='pdf')
    plt.show()

def plottingma2(m):
    xlen = range(len(m))
    ax = plt.gca()
    plt.plot(xlen, m[:,1], "-", label=r'$tap_1$')
    plt.plot(xlen, m[:,2], "--", label=r'$tap_2$')

    plt.ylim((0,2))
    plt.xlabel(r"Length of signal $(\times 10^4)$")
    plt.ylabel("Ratio between standard deviation and expectation")
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc=0, ncol=2, fontsize=15)
    plt.grid()


if __name__ == "__main__":
    filename = "convergence_cx3_long_cumulant_pcs123.txt"
    result = calculate(filename)

    filename = re.sub(".txt","",filename)+".pdf"
    matcher = re.compile("long|short|cx|ma")
    matched = matcher.findall(filename)
    if "long" in matched and "cx" in matched:
        plottingcx11(percentage(result))
    elif "short" in matched and "cx" in matched:
        plottingcx5(percentage(result))
    elif "long" in matched and "ma" in matched:
        plottingma5(percentage_3row(result))
    elif "short" in matched and "ma" in matched:
        plottingma2(percentage_3row(result))
    else:
        print "You might need to add more plotting functions..."
    plt.savefig(filename, format='pdf')
    plt.show()

