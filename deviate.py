import numpy as np
from numpy import log
rnormal = np.random.normal
uniform = np.random.uniform
randint = np.random.randint
exponential = np.random.exponential

# Centered normal random deviate
normal_deviate = lambda var : rnormal(0,var)

# Centered uniform random deviate
uniform_deviate = lambda half_width: uniform(-half_width, half_width)

# Centered discrete uniform random deviate
discrete_uniform_deviate = lambda half_width: randint(-half_width, half_width)

def double_exponential_deviate(beta):
    """Centered double-exponential random deviate"""
    u = random_number()
    if u<0.5:
        return beta*log(2*u)
    return -beta*log(2*(1-u))

# create "Independent exponential distributed random deviates with mean one"
def generater_expo(beta, size):
    return exponential(beta, size)-beta

def save_signal(beta, nmc):
    # generate enough length of random data with given distribution
    length = 7*8*9*11*128
    for i in range(nmc):
        signal = generater_expo(beta,length)
        np.save("data/exp_deviate_one_%d.npz"%(i), signal)
        print "Save data for the %d round of monte carlo."%(i)

if __name__=="__main__":
    save_signal(1, 50)

