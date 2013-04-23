import re
import numpy as np

finder = re.compile("-?\d+.\d*")
f = open("test_result.txt", 'r')
result3 = []
result4 = []
for line in f:
    temp = [float(k) for k in finder.findall(line)]
    if temp == []: break
    result3.append(temp)
for line in f:
    temp = [float(k) for k in finder.findall(line)]
    if temp == []: continue
    result4.append(temp)

for i in range(1,6):
    f = open("test%s/test_result.txt"%(i), 'r')
    for line in f:
        temp = [float(k) for k in finder.findall(line)]
        if temp == []: break
        result3.append(temp)
    for line in f:
        temp = [float(k) for k in finder.findall(line)]
        if temp == []: continue
        result4.append(temp)



np.save("overall_3rd_order_cumulant.npy", np.array(result3))
np.save("overall_4rd_order_cumulant.npy", np.array(result4))


print result3
print "\n\n", result4


