import numpy as np
from scipy.linalg import circulant


def display(n=10):
    s1 = [n-i for i in range(n)]+[0 for i in range(n-1)]
    # build causal right circulant matrix
    # np.matrix rather than np.ndarray could perform matrix mulitplication
    s1 = np.matrix(np.flipud(np.fliplr(circulant(s1)[n-1:,:])))
    #s2 = np.matrix(s1).T
    # using a diffrent and random matrix for generality
    s2 = [np.random.randint(1,20) for i in range(n)]+[0 for i in range(n-1)]
    s2 = np.matrix(np.flipud(np.fliplr(circulant(s2)[n-1:,:]))).T

    print s1
    print "\n", s2
    print "\n", s1*s2

if __name__ == "__main__":
    display()
