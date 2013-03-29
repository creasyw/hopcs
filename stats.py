import matplotlib.pyplot as plt
import numpy as np

nmc = 50
diff = 102
care = 50

def plot_b1():
    bf1 = {}
    bc1 = {}
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel("Frame number", size=15)
    ax.set_ylabel("Esimated value", labelpad=15, size=15)
    
    fig.add_subplot(311)
    plt.subplots_adjust(hspace=0.5)
    plt.grid()
    plt.title("Original MA estimation")
    for i in range(0,nmc):
        bf1[i]= np.load("result/exp_deviate_b1_full_%d.npy"%(i))
        bf1[i] = np.array([bf1[i][k] for k in range(len(bf1[i])) if k%diff==0])[:care]
        plt.plot(bf1[i])
    
    fig.add_subplot(312)
    plt.title("PCS MA estimation")
    plt.ylim(-500,200)
    plt.grid()
    for i in range(0, nmc):
        bc1[i]= np.load("result/exp_deviate_b1_cp_%d.npy"%(i))
        bc1[i] = bc1[i][:care]
        plt.plot(bc1[i])
    
    # average
    bf1_av = reduce(lambda x, y: x+y, bf1.values(), np.zeros(len(bf1[0])))/nmc
    bc1_av = reduce(lambda x, y: x+y, bc1.values(), np.zeros(len(bc1[0])))/nmc
    fig.add_subplot(313)
    plt.grid()
    plt.plot(bf1_av)
    plt.plot(bc1_av)
    plt.title("Averaging Monte Carlo output")
    
    plt.savefig("b1.pdf")
    plt.show()

def plot_b2():
    bf2 = {}
    bc2 = {}
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel("Frame number", size=15)
    ax.set_ylabel("Esimated value", labelpad=15, size=15)
    
    fig.add_subplot(311)
    plt.subplots_adjust(hspace=0.5)
    plt.grid()
    plt.title("Original MA estimation")
    for i in range(0,nmc):
        bf2[i]= np.load("result/exp_deviate_b2_full_%d.npy"%(i))
        bf2[i] = np.array([bf2[i][k] for k in range(len(bf2[i])) if k%diff==0])[:care]
        plt.plot(bf2[i])
    
    fig.add_subplot(312)
    plt.title("PCS MA estimation")
    plt.ylim(-500,200)
    plt.grid()
    for i in range(0, nmc):
        bc2[i]= np.load("result/exp_deviate_b2_cp_%d.npy"%(i))
        bc2[i] = bc2[i][:care]
        plt.plot(bc2[i])
    
    # average
    bf2_av = reduce(lambda x, y: x+y, bf2.values(), np.zeros(len(bf2[0])))/nmc
    bc2_av = reduce(lambda x, y: x+y, bc2.values(), np.zeros(len(bc2[0])))/nmc
    fig.add_subplot(313)
    plt.grid()
    plt.plot(bf2_av)
    plt.plot(bc2_av)
    plt.title("Averaging Monte Carlo output")
    
    plt.savefig("b2.pdf")
    plt.show()

if __name__=="__main__":
    plot_b2()
