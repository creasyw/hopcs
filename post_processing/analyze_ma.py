#! /usr/bin/python

import re
import sys, os
import numpy as np

short_taps = np.array([1, -2.333, 0.667])
long_taps = np.array([1, 0.1, -1.87, 3.02, -1.435, 0.49])

def rmse(estimate):
    lags = len(estimate[0])
    length = len(estimate)
    if lags == 3:
        benchmark = short_taps
    elif lags == 6:
        benchmark = long_taps
    else:
        raise Exception("The length of data is incorrect.")
    return np.sqrt(sum(map(lambda x: (x-benchmark)**2, estimate))/length)


def main():
    finder = re.compile("\[|-?\d+\.\d*[eE]?[-+]?\d*|\]")

    filename = sys.argv[-1]
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

    result = np.array(filter(lambda k: max(np.abs(k))<5, result))
    print "\n", filename
    print np.mean(result, 0)
    print np.var(result,0)
    print rmse(result)
    return result


def test(result):
    tap1 = result[:,1]
    tap2 = result[:,2]
    threshold = 20

    mtap1 = np.median(tap1)
    mtap2 = np.median(tap2)
    tap1 = filter(lambda x: abs(x-mtap1)<10, tap1)
    tap2 = filter(lambda x: abs(x-mtap2)<10, tap2)
    print len(tap1), len(tap2)
    print np.mean(tap1), np.var(tap1), np.mean(tap2), np.var(tap2)


if __name__ == "__main__":
    main()

