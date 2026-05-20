import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LotkaVolterraSimulator:
    """
    Simulates the Lotka-Volterra predator-prey model
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, delta=0.1):
        self.alpha = alpha  # prey growth rate
        self.beta = beta    # prey death rate
        self.gamma = gamma  # predator death rate
        self.delta = delta  # predator growth rate
    
    def simulate(self, x0, t_span, n_steps=100):
        """
        Simulate the system
        x0: initial state [prey, predator]
        t_span: time span
        """
        def lv_model(t, y):
            prey, predator = y
        return [self.alpha * prey - self.beta * prey * predator,
                self.delta * prey * predator - self.gamma * predator]
        
        # Solve ODE
        solution = solve_ivp(lv_model, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], n_steps))
        
        return solution.t, solution.y

class SIRDSimulator:
    """
    Simulates the SIRD disease model
    """
    
    def __init__(self, beta=0.1, gamma=0.05, mu=0.01):
        self.beta = beta    # infection rate
        self.gamma = gamma  # recovery rate
        self.mu = mu        # death rate
    
    def simulate(self, x0, t_span, n_steps=100):
        """
        Simulate the system
        x0: initial state [susceptible, infected, recovered, deceased]
        t_span: time span
        """
        def sird_model(t, y):
            s, i, r, d = y
        return [-self.beta * s * i,
                self.beta * s * i - self.gamma * i - self.mu * i,
                self.gamma * i,
                self.mu * i]
        
        # Solve ODE
        solution = solve_ivp(sird_model, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], n_steps))
        
        return solution.t, solution.y

class HodgkinHuxleySimulator:
    """
    Simulates the Hodgkin-Huxley neuron model
    """
    
    def __init__(self, Cm=1.0, gNa=120, gK=36, gL=0.03, EL=-65, ENa=50, EK=-77):
        self.Cm = Cm
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.EL = EL
        self.ENa = ENa
        self.EK = EK
    
    def simulate(self, x0, t_span, n_steps=100):
        """
        Simulate the system
        x0: initial state [voltage, m, h, n]
        t_span: time span
        """
        def hh_model(t, y):
            v, m, h, n = y
        # Rate functions
        alpha_m = 0.1 * (v + 40) / (1 - np.exp(-0.1 * (v + 40)))
        beta_m = 4 * np.exp(-0.1 * (v + 40))
        alpha_h = 0.07 * np.exp(-0.1 * (v + 40))
        beta_h = 1 / (1 + np.exp(-0.1 * (v + 40)))
        alpha_n = 0.01 * (v + 40) / (1 - np.exp(-0.1 * (v + 40)))
        beta_n = 0.125 * np.exp(-0.1 * (v + 40))
        
        # Currents
        INa = self.gNa * m**3 * h * (v - self.ENa)
        IK = self.gK * n**4 * (v - self.EK)
        IL = self.gL * (v - self.EL)
        
        # Derivatives
        dvdt = (1/self.Cm) * (INa + IK + IL)
        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        dndt = alpha_n * (1 - n) - beta_n * n
        
        return [dvdt, dmdt, dhdt, dndt]
        
        # Solve ODE
        solution = solve_ivp(hh_model, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], n_steps))
        
        return solution.t, solution.y

def main():
    """
    Main function to run simulations
    """
    print("Running simulations...")
    
    # Run Lotka-Volterra simulation
    print("Running Lotka-Volterra simulation...")
    lv_sim = LotkaVolterraSimulator(alpha=1.0, beta=0.1, gamma=0.1, delta=0.1)
    t, x = lv_sim.simulate([10, 5], [0, 10], n_steps=100)
    print("Lotka-Volterra simulation completed")
    
    # Run SIRD simulation
    print("Running SIRD simulation...")
    sird_sim = SIRDSimulator(beta=0.1, gamma=0.05, mu=0.01)
    t, x = sird_sim.simulate([100, 1, 0, 0], [0, 10], n_steps=100)
    print("SIRD simulation completed")
    
    # Run Hodgkin-Huxley simulation
    print("Running Hodgkin-Huxley simulation...")
    hh_sim = HodgkinHuxleySimulator()
    t, x = hh_sim.simulate([-65, 0, 0, 0], [0, 10], n_steps=100)
    print("Hodgkin-Huxley simulation completed")
    
    print("All simulations completed!")

if __name__ == "__main__":
    main()