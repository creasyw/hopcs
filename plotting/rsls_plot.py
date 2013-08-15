import matplotlib.pyplot as plt
import sys, os, re
import numpy as np

# The following script is specifically for the throughput output file
finder = re.compile("\d+.\d*")
#assert len(sys.argv)==2, "The format of input should be $python plot.py filename"

filename = sys.argv[1]
savename = sys.argv[2]
thp = []
with open(os.path.join(os.path.dirname(__file__), filename)) as filelist:
    for name in filelist:
        temp = []
        name = re.sub("\n","", name)
        with open(os.path.join(os.path.dirname(__file__), name)) as data:
            for row in data:
                finding = finder.findall(row)
                if len(finding) != 0:
                    temp.append(float(finding[0]))
        temp = np.array(sorted(temp))
        thp.append(temp)


# post-processing
thp = np.array(thp)
result = []
step = 5
print thp.shape
for i in range(4):
  result.append(sum(thp[i*step:(i*step+step),:])/float(step))

plt.plot(result[0], np.arange(1,len(result[0])+1)/float(len(result[0])), 'r-.', label="JTC")
plt.plot(result[1], np.arange(1,len(result[1])+1)/float(len(result[0])), 'b--', label="SCM_LOS")
plt.plot(result[2], np.arange(1,len(result[2])+1)/float(len(result[0])), 'g:', label="SCM")
plt.plot(result[3], np.arange(1,len(result[3])+1)/float(len(result[0])), 'k-', label="WINNER2")
plt.legend(loc=0)
plt.savefig(savename, format='pdf')
plt.show()

