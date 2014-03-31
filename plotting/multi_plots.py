import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import sys, os
import numpy as np
from variance import calculate, to_percent, percentage
import matplotlib.cm

def ratio_plot(mns, pcs, filename):
    fig = plt.figure()
    xlen = range(len(mns))
    
    #-- Panel 1
    ax1 = fig.add_subplot(511)
    ax1.stackplot(xlen, mns[:,0]/pcs[:,0], colors=['gray'])
    ax1.set_ylabel(r'$c_{3y}(-2,0)$')
    ax1.locator_params(axis='y', tight=True, nbins=3)
    ax1.grid()

    #-- Panel 2
    ax2 = fig.add_subplot(512, sharex=ax1)
    ax2.stackplot(xlen, mns[:,1]/pcs[:,1], colors=['gray'])
    ax2.set_ylabel(r'$c_{3y}(-1,0)$')
    ax2.locator_params(axis='y', tight=True, nbins=3)
    ax2.grid()

    #-- Panel 3
    ax3 = fig.add_subplot(513, sharex=ax1)
    ax3.stackplot(xlen, mns[:,2]/pcs[:,2], colors=['gray'])
    ax3.set_ylabel(r'$c_{3y}(0,0)$')
    ax3.locator_params(axis='y', tight=True, nbins=3)
    ax3.grid()

    ax4 = fig.add_subplot(514, sharex=ax1)
    ax4.stackplot(xlen, mns[:,3]/pcs[:,3], colors=['gray'])
    ax4.set_ylabel(r'$c_{3y}(1,0)$')
    ax4.locator_params(axis='y', tight=True, nbins=3)
    ax4.grid()

    ax5 = fig.add_subplot(515, sharex=ax1)
    ax5.stackplot(xlen, mns[:,4]/pcs[:,4], colors=['gray'])
    ax5.set_ylabel(r'$c_{3y}(2,0)$', labelpad=15)
    ax5.locator_params(axis='y', tight=True, nbins=2)
    ax5.grid()

    ax1.xaxis.set_ticks(range(10,70,10))
    plt.xlabel(r"Length of signal $(\times 10^4)$", labelpad=15)

    # Remove space between subplots
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(filename, format='pdf')
    plt.show()

def ratio():
    f1 = "convergence_cx3_short_cumulant_mns434.txt"
    f2 = "convergence_cx3_short_cumulant_pcs235.txt"
    mns = percentage(calculate(f1))
    pcs = percentage(calculate(f2))
    ratio_plot(mns, pcs, "cx3_ratio.pdf")


if __name__=="__main__":
    ratio()
    
