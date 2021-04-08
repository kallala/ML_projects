
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()
plt.show()
np.random.seed(1552)
dims = 2
alpha=5.
w = np.random.randn(dims,1)*np.sqrt(alpha**-1)
N = 10**5
xmin = -10; xmax=10.
X = np.zeros((N,dims))
X[:,1] = (np.random.rand(N)-0.5)*(xmax-xmin)
X[:,0] = 1.

sig=10
t = X@w+np.random.randn(N,1)*sig

w_i = np.zeros((2,1))
batch_size = 1000
n_it = int(N/batch_size)
n_epochs = 10

S0 = np.eye(2)*alpha**-1
M0 = np.zeros((2,1))

n_prev = 0
estimated_beta_inv = 0
err = []
for j in range(n_epochs):
    print("epoch",j+1,"/",n_epochs)
    for i in tqdm(range(n_it)):
        #imin = i * batch_size
        #imax = imin+batch_size
        rang = np.random.randint(0,N,size=batch_size)
        batch_x = X[rang,:]
        batch_t = t[rang,:]
        new_beta_inv_estimate= 1./batch_size*np.sum((batch_t-batch_x@w_i)**2,axis=0)
        estimated_beta_inv = estimated_beta_inv*n_prev/(n_prev+batch_size) + batch_size/(n_prev+batch_size)*new_beta_inv_estimate
        n_prev=n_prev+batch_size
        #estimated_beta_inv = 1./N *np.sum((t-X@w_i)**2,axis=0)
        estimated_beta = 1./estimated_beta_inv

        Sn = np.linalg.inv(np.linalg.inv(S0) + estimated_beta*batch_x.T@batch_x)
        Mn = Sn@(np.linalg.inv(S0)@M0 + estimated_beta*batch_x.T@batch_t )
        S0 = Sn*1
        M0 = Mn*1
        w_i = Mn
        err.append(np.sqrt(np.sum(abs(w_i-w)**2)))

estimated_sig = estimated_beta_inv
plt.plot(err)
plt.yscale("log")
