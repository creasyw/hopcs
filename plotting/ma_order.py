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


def hist_plot_one_y(m, n, p, m1, n1, p1, filename):
    x = np.arange(0,len(m))
    common_params = dict(bins=len(m), range=(0,len(m)), normed=False)
    from pylab import *
    # The original setting is 8,6
    # lookup via "matplotlib.rcParams.values"
    rcParams['figure.figsize'] = 8, 8

    ax1 = plt.subplot(211)
    plt.hist((x,x,x), weights=(m,n,p), **common_params)
    names = ["lag#0","lag#1","lag#2","lag#3","lag#4","lag#5","lag#6"]
    ax1.xaxis.label.set_fontsize(15)
    ax1.yaxis.label.set_fontsize(15)
    ax1.set_xticks([k+0.5 for k in range(7)])
    #ax.set_xticklabels(names,rotation=30, rotation_mode="anchor", ha="right")
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Amplitude of cumulants", labelpad=10)
    ax1.set_xlabel("Lags of the model", labelpad=10)
    plt.tight_layout()
    plt.grid(axis='y')

    ax1 = plt.subplot(212)
    plt.hist((x,x,x), weights=(m1,n1,p1), **common_params)
    names = ["lag#0","lag#1","lag#2","lag#3","lag#4","lag#5","lag#6"]
    ax1.xaxis.label.set_fontsize(15)
    ax1.yaxis.label.set_fontsize(15)
    ax1.set_xticks([k+0.5 for k in range(7)])
    #ax.set_xticklabels(names,rotation=30, rotation_mode="anchor", ha="right")
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Amplitude of cumulants", labelpad=10)
    ax1.set_xlabel("Lags of the model", labelpad=10)
    plt.tight_layout()
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def hist_plot_two_y(m, n, p, mv, nv, pv, filename):
    x = np.arange(0,len(m))
    ax = plt.subplot(111)
    common_params = dict(bins=len(m), range=(0,len(m)), normed=False)

    plt.hist((x,x,x), weights=(m,n,p), **common_params)

    ax2 = plt.twinx()
    ax2.plot(map(lambda i:i+0.5, x), np.array(mv)/np.array(m), linestyle=':', color='b', marker='o', linewidth=1)
    ax2.plot(map(lambda i:i+0.5, x), np.array(nv)/np.array(n), linestyle='-.', color='g', marker='o', linewidth=1)
    ax2.plot(map(lambda i:i+0.5, x), np.array(pv)/np.array(p), linestyle='-', color='r', marker='o', linewidth=1)
    formatter = FuncFormatter(to_percent)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.yaxis.label.set_fontsize(15)
    ax2.set_ylabel("Ratio between variance and amplitude", labelpad=10)

    names = ["lag#0","lag#1","lag#2","lag#3","lag#4","lag#5","lag#6"]
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    ax.set_xticks([k+0.5 for k in range(7)])
    #ax.set_xticklabels(names,rotation=30, rotation_mode="anchor", ha="right")
    ax.set_xticklabels(names)
    ax.set_ylabel("Amplitude of cumulants", labelpad=10)
    ax.set_xlabel("Lags of the model", labelpad=10)
    plt.tight_layout()
    plt.grid(axis='y')

    plt.savefig(filename, format='pdf')
    plt.show()

def nonoise_and_gaussian():
    cm2 = [2.52949481, 0.936120165, 0.313697319,  0.767163541,  0.00595326677, 0.00182812194, 0.00188182827]
    cm3 = [2.58890832,  2.14882287, 1.36858924, 1.55519796, 0.01710334,  0.01816029, 0.02361964]
    cx3 = [2.67203548, 2.19597631, 1.38880874, 1.57576994, 0.01724148, 0.02563635, 0.02521171]
    
    cm2g = [4.79871303, 0.94108659, 0.32095805, 0.75577782, 0.01591854, 0.01455689, 0.00556109]
    cm3g = [2.68425868, 2.15788692, 1.35342272, 1.60591685, 0.01381513, 0.00614096, 0.02954263]
    cx3g = [2.30306421, 2.0669669, 1.535906, 2.2561455, 0.69102309,  0.45103686, 0.0109114 ]

    hist_plot_one_y(cm2, cm3, cx3, cm2g, cm3g, cx3g, "amplitude_works.pdf")

def color_gaussian():
    cm2 = [17.90646902, 5.61932575, 1.84897127, 2.75154841, 5.86796282, 2.40462868, 0.06414782]
    cm3 = [3.91778573, 2.11276639, 1.1144655, 2.2602607, 0.4291346, 0.13745781, 0.82875333]
    cx3 = [3.50543424, 2.59394735, 1.53593361, 2.25595103, 0.44131118, 0.25054655, 0.54635108]
    
    cm2v = [0.28242889, 0.06305669, 0.11931309, 0.0370247, 0.13120133, 0.07117453, 0.15179621]
    cm3v = [7.16843174, 3.25228882, 1.10016263, 1.53312499, 2.47880261, 1.16758318, 1.78114194]
    cx3v = [7.15984781, 1.74769222, 1.71143783, 2.60130217, 2.54000828, 2.07532114, 3.61305697]

    hist_plot_two_y(cm2, cm3, cx3, cm2v, cm3v, cx3v, "color_gaussian.pdf")


if __name__ == "__main__":
    #nonoise_and_gaussian()
    color_gaussian()

