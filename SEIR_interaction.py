import numpy as np
from ODESolver import RungeKutta4
import matplotlib.pyplot as plt
class Region:
    def __init__(self, name, S_0 ,E2_0):
        self.E1_0 = 0
        self.name = name
        self.S_0 = S_0
        self.E2_0 = E2_0
        self.I_0 = 0
        self.Ia_0 = 0
        self.R_0 = 0
        self.t = 0
        self.population = self.S_0 + self.E2_0

    def set_SEIR_values(self, u, t):

        self.S_0 = u[:,0]
        self.E1_0 = u[:,1]
        self.E2_0 = u[:,2]
        self.I_0 = u[:,3]
        self.Ia_0 = u[:,4]
        self.R_0 = u[:,5]
        self.t = t

    def plot(self):
        plt.plot(self.t, self.S_0, label='S(t)')
        plt.plot(self.t, self.I_0, label='I(t)')
        plt.plot(self.t, self.Ia_0, label='Ia(t)')
        plt.plot(self.t, self.R_0, label='R(t)')
        plt.grid()
        plt.legend()


class ProblemSEIR:
    def __init__(self, region, beta, r_ia=0.1, r_e2=1.25, lmbda_1=0.33,
                 lmbda_2=0.5, p_a=0.4, mu=0.2):
        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta
        elif callable(beta):
            self.beta = beta
        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu
        self.region = region
        self.set_initial_condition()



    def set_initial_condition(self):
        region = self.region
        self.initial_condition = [region.S_0, region.E1_0, region.E2_0, region.I_0, region.Ia_0, region.R_0]

    def get_population(self):
        return self.region.population

    def solution(self,u,t):
        return self.region.set_SEIR_values(u, t)

    def __call__(self,u,t):

        S, E1, E2, I, Ia, R = u
        print(u)
        N = sum(u)
        dS = -self.beta(t) * S * I / N - self.r_ia * self.beta(t) * S * Ia / N - self.r_e2 * self.beta(t) * S * E2 / N
        dE1 = self.beta(t) * S * I / N + self.r_ia * self.beta(t) * S * Ia / N + self.r_e2 * self.beta(t) * S * E2 / N - self.lmbda_1 * E1
        dE2 = self.lmbda_1 * (1 - self.p_a) * E1 - self.lmbda_2 * E2
        dI = self.lmbda_2 * E2 - self.mu * I
        dIa = self.lmbda_1 * self.p_a * E1 - self.mu * Ia
        dR = self.mu * (I + Ia)

        return [dS, dE1, dE2, dI, dIa, dR]


class SolverSEIR:
    def __init__(self, problem, T, dt):
        self.problem = problem  # instance of class ProblemSEIR
        self.T = T  # final time
        self.dt = dt
        self.total_population = self.problem.get_population

    def solve(self, method=RungeKutta4):
        solver = method(self.problem)
        solver.set_initial_condition([5e6 , 0 , 100 , 0 , 0 , 0])
        # calculate the number of time steps from T and dt
        N = int(self.T/self.dt)
        t = np.linspace(0, self.T, N)
        u, t = solver.solve(t)
        self.problem.region.set_SEIR_values(u, t)
        self.problem.solution(u, t)


class RegionInteraction(Region):
    def __init__(self,name,S_0, E2_0,lat, long):
        super().__init__(name,S_0, E2_0)
        self.lat = lat*(np.pi/180)
        self.long = long * (np.pi/180)
    def distance(self, other):
        return np.arccos(np.sin(self.lat)*np.sin(other.lat) +
                         np.cos(self.lat)*np.cos(other.lat)*
                         np.cos(abs(self.long - other.long)))*64

class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia = 0.1, r_e2=1.25,\
                 lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        self.region=region
        self.area_name=area_name
        super().__init__(region,beta, r_ia = 0.1, r_e2=1.25,\
                lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2)

    def get_population(self):
        s = 0
        for i in range(len(self.region)):
            s += self.region[i].population
        return s

    def set_initial_condition(self):
        self.initial_condition = []
        for i in range(len(self.region)):
            self.initial_condition = self.set_initial_condition + self.region[i]
        return self.initial_condition

    def __call__(self, u, t):
        n = len(self.region)

        SEIR_list = [u[i:i + 6] for i in range(0, len(u), 6)]

        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]

        derivative = []
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            N = S + E1 + E2 + I + Ia + R
            dS = 0
            dE1 = 0
            dE2 = 0
            dI = 0
            dIa = 0
            dR = 0
            for j in range(n):
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                N_j = self.region[j].population
                dij = self.region[i].distance(self.region[j])
                dS += -self.beta(t) * S * I / N - self.r_ia * self.beta(t) * S * Ia_other / N_j - self.r_e2 * self.beta(
                    t) * S * (E2_other / N_j) * np.exp(-dij)
                dE1 = -dS - self.lmbda_1 * E1
                dE2 = self.lmbda_1 * (1 - self.p_a) * E1 - self.lmbda_2 * E2
                dI = self.lmbda_2 * E2 - self.mu * I
                dIa = self.lmbda_1 * self.p_a * E1 - self.mu * Ia
                dR = self.mu * (I + Ia)
                derivative = [dS, dE1, dE2, dI, dIa, dR]
        return derivative
if __name__ == '__main__':
    innlandet = RegionInteraction('Innlandet',S_0=371385, E2_0=0, \
                         lat=60.7945,long=11.0680)
    oslo = RegionInteraction('Oslo',S_0=693494,E2_0=100, \
                         lat=59.9,long=10.8)

    print(oslo.distance(innlandet))

    # problem = ProblemInteraction([oslo,innlandet],'Norway_east', beta=0.5)
    # print(problem.get_population())
    # problem.set_initial_condition()
    # print(problem.initial_condition) #non-nested list of length 12
    # u = problem.initial_condition
    # print(problem(u,0)) #list of length 12. Check that values make sense

    #when lines above work, add this code to solve a test problem:
    # solver = SolverSEIR(problem,T=100,dt=1.0)
    # solver.solve()
    # problem.plot()
    # plt.legend()
    # plt.show()

"""
Run example:

user$ python3 SEIR_interaction.py

output: 1.0100809386285283

This output makes sense as its about 100 km from Oslo to Innlandet.

The rest of this code is broken due to my lack of knowledge, sorry :)
"""
