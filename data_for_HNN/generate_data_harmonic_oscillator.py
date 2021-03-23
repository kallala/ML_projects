import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm

class oscillator():
    def __init__(self, **kwargs):
        self.m = kwargs.get("m",1.)
        self.k = kwargs.get("k",1.)
        self.omega0 = (self.k/self.m)**0.5
    def compute_hamiltonian(self, trajectory):
        hamiltonian =  trajectory[0,:]**2*self.k +trajectory[1,:]**2/self.m
        hamiltonian *= 0.5
        return hamiltonian
    def compute_hamiltonian_derivative(self,t,x):
        f0 = x[1]/self.m
        f1 = -self.k*x[0]
        res = np.array([f0,f1])
        return res
    def simulate(self, x0, dt, n_steps, method = "RK45"):
        tspan = np.arange(n_steps)*dt
        sim_result = scipy.integrate.solve_ivp(fun = self.compute_hamiltonian_derivative, method= method,
                                              y0=x0, t_span = [tspan.min(),tspan.max()], t_eval= tspan, vectorized=True)
        self.orbit = sim_result.y
        self.t = tspan
        self.exact_solution = sim_result.y*0
        A = x0[0]; B = x0[1]/(self.k)
        self.exact_solution[0,:] = A*np.cos(self.omega0*self.t)+B*np.sin(self.omega0*self.t)
        self.exact_solution[1,:] = (-A*np.sin(self.omega0*self.t)+B*np.cos(self.omega0*self.t))*self.omega0
def main():
    kwargs = {}
    kwargs["m"]=1
    kwargs["k"] = 2
    osc = oscillator(**kwargs)

    n_sims = 100
    nstep =10000

    x0 = np.random.randn(n_sims*2)*10
    x0=np.reshape(x0,(2,n_sims))
    dt = (2*np.pi/osc.omega0)/(nstep+1)
    for i in tqdm(range(n_sims)):
        n_step = int(2*np.pi/osc.omega0/dt)+1
        osc.simulate(x0[:,i], dt=dt, n_steps=nstep, method = "RK45") #, method = "DOP853")
    #    integrated_energy = osc.compute_hamiltonian(osc.orbit)
        exact_energy =  osc.compute_hamiltonian(osc.exact_solution)
        if(i==0):
            trajectories = osc.exact_solution.T
            energies = exact_energy*1
        else:
            trajectories = np.concatenate((trajectories,osc.exact_solution.T))
            energies = np.concatenate((energies,exact_energy))
    x_train = trajectories
    y_train = 0*x_train
    y_train[:,0] = x_train[:,0]*osc.k
    y_train[:,1] = (1/osc.m)*x_train[:,1]
    np.save("x_train_harmonic_oscillator", x_train)
    np.save("y_train_harmonic_oscillator", y_train)
if __name__ == "__main__":
    main()
