import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

L=1.
g=9.8
m=1.
def func( t, y ):
    global g,L

    return [ y[1], -g/L * np.sin( y[0] ) ]
def eval_derivatie_hamiltonian(sim_result):
    global m,g,L
    nsteps=sim_result.shape[1]
    result = np.zeros((2,nsteps))
    result[0,:] = sim_result[1,:]/m*L**2
    result[1,:] = np.sin(sim_result[0,:])*m*g*L
    return result
def main():
    n_sims = 100
    nsteps =10000
    theta0 = np.random.rand(n_sims)*2*np.pi
    theta_dot0 = 2*(np.random.rand(n_sims)-0.5)*3
    omega_pendulum= np.sqrt(g/L)
    T_pendulum = 2*np.pi/omega_pendulum

    t_max = 10*T_pendulum
    for i in tqdm(range(n_sims)):
        t_eval = np.linspace(0,t_max,nsteps)# np.random.rand(nsteps)*t_max
        #t_eval = np.sort(t_eval)
        x0 = np.array([theta0[i],theta_dot0[i]])
        sim_result = scipy.integrate.solve_ivp(fun = func, method= "RK45",
                                              y0=x0, t_span = [0,t_max], t_eval= t_eval, vectorized=True).y
        sim_result[:,1]*= m*L**2
        H_prime = eval_derivatie_hamiltonian(sim_result)
        if(i==0):
            x_train =sim_result.T
            y_train = H_prime.T
        else:
            x_train = np.concatenate((x_train,sim_result.T))
            y_train =  np.concatenate((y_train,H_prime.T))
    np.save("x_train_pendulum", x_train)
    np.save("y_train_pendulum", y_train)

if __name__ == "__main__":
    main()
