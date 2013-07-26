import re
import sys, os
import numpy as np

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

if __name__=="__main__":
    rms = calculte("convergence_ma3_long_cumulant_pcs123.txt")
    for i in range(len(rms[0])):
        print min(rms[:,i])
    
    print rms[-1,:]

