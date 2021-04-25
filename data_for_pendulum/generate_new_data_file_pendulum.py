import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class simplectic_integrator():
    def __init__(self,order=4):
        self.order = order
        if self.order == 1:
            c = np.array([1.],dtype=np.float64)
            d = np.array([1.],dtype=np.float64)
        if self.order==2:
            c = np.array([0,1],dtype=np.float64)
            d = np.array([0.5,0.5],dtype=np.float64)
        if self.order==3:
            c = np.array([1.,-2/3,2/3],dtype=np.float64)
            d = np.array([-1/24.,3./4.,7./24],dtype=np.float64)
        if self.order==4:
            c1 = 1./(2*(2-2**(1/3)))
            c2 = (1-2**(1/3))/(2*(2-2**(1/3)))
            c = np.array([c1,c2 , c2 ,c1 ],dtype=np.float64)
            d1 = 1./(2-2**(1/3))
            d2 = - 2**(1/3)/(2-2**(1/3))
            d4 = 0
            d = np.array([d1,d2,d1,d4], dtype=np.float64)
        self.c = c
        self.d = d
class pendulum():
    def __init__(self, g, L,m, dt, order, init_pos, init_mom):
        self.g = g
        self.L = L
        self.dt = dt
        self.m=m
        self.current_position = init_pos
        self.current_momentum = init_mom
        self.order = order
        self.simplectic_coeffs = simplectic_integrator(self.order)
        self.d = self.simplectic_coeffs.d
        self.c = self.simplectic_coeffs.c
        self.Force = lambda x : -self.g/self.L*np.sin(x)

    def one_step_integration(self):
        for i in range(self.order):
            self.current_momentum +=self.dt*self.d[i]*self.Force(self.current_position)
            self.current_position +=self.dt*self.c[i]*self.current_momentum

    def compute_hamiltonian(self,pos, mom):
        H = self.L**2/2*mom**2+self.g*self.L*(1-np.cos(pos))
        return H
    def simplectic_integrator(self, nsteps,  init_pos,init_mom):
        result = np.zeros((nsteps+1,2),dtype=np.float64)
        self.current_position = init_pos
        self.init_mom=init_mom
        result[0,:] = np.array([self.current_position,self.current_momentum])

        for i in (range(nsteps)):
            self.one_step_integration()

            result[1+i,:] = np.array([self.current_position,self.current_momentum])
        return result

    def eval_derivatie_hamiltonian(self, sim_results):

        result = 0*sim_results
        result[:,0] = sim_results[:,1]
        result[:,1] = np.sin(sim_results[:,0])*self.m*self.g*self.L
        return result
    def eval_theta_p(self,sim_results):
        result =0*sim_results
        result[:,0]  = sim_results[:,0]
        result[:,1] = sim_results[:,1]*self.m*self.L**2
        return result

n_sims = 100
nsteps=10000
dt = 1e-3
mysystem=pendulum(g=9.8,L=1,m=1, dt=dt,order=4, init_pos=0, init_mom=0)

for i in tqdm(range(n_sims)):
    init_pos = np.random.rand()*2*np.pi
    init_mom = 0# np.random.rand()-0.5
    result =mysystem.simplectic_integrator(nsteps,init_pos,init_mom)
    if(i==0):
        all_result=result*1
    else:
        all_result = np.vstack((all_result,result))

x_train = mysystem.eval_theta_p(all_result)
y_train = mysystem.eval_derivatie_hamiltonian(all_result)
np.save("x_train_pendulum", x_train)
np.save("y_train_pendulum", y_train)
