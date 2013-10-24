import numpy as np
from scipy.linalg import circulant


def display(n=10):
    s1 = [n-i for i in range(n)]+[0 for i in range(n-1)]
    # build causal right circulant matrix
    # np.matrix rather than np.ndarray could perform matrix mulitplication
    s1 = np.matrix(np.flipud(np.fliplr(circulant(s1)[n-1:,:])))
    s2 = np.matrix(s1)

if __name__ == "__main__":
    display()
