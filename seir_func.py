import numpy as np
import matplotlib.pyplot as plt
def SEIR(u, t):
    beta = 0.5
    r_ia = 0.1
    r_e2 = 1.25
    lmbda_1 = 0.33
    lmbda_2 = 0.5
    p_a = 0.4
    mu = 0.2

    S, E1, E2, I, Ia, R = u
    N = sum(u)
    dS = -beta * S * I / N - r_ia * beta * S * Ia / N - r_e2 * beta * S * E2 / N
    dE1 = beta * S * I / N + r_ia * beta * S * Ia / N + r_e2 * beta * S * E2 / N - lmbda_1 * E1
    dE2 = lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2
    dI = lmbda_2 * E2 - mu * I
    dIa = lmbda_1 * p_a * E1 - mu * Ia
    dR = mu * (I + Ia)
    return [dS, dE1, dE2, dI, dIa, dR]


def test_SEIR():
    t = 0
    u = [1, 1, 1, 1, 1, 1]
    computed = SEIR(u, t)
    tol = 1e-10
    expected = [-0.19583333333333333, -0.13416666666666668, -0.302, 0.3, -0.068, 0.4]
    tol = 1e-10
    for x, exp in zip(computed, expected):
        assert abs(x - exp) < tol, \
            f'Failed for x = {x}, expected {exp}, but got {f(x)}'


from ODESolver import *


def solve_SEIR(T, dt, S_0, E2_0):
    time_points = np.linspace(0, T)
    solver = RungeKutta4(SEIR)
    solver.set_initial_condition([S_0, 0, E2_0, 0, 0, 0])
    u,t = solver.solve(time_points)
    return u,t

def plot_SEIR(u,t):
    plt.plot(t, u[:,0], label = 'S(t)')
    plt.plot(t, u[:,3], label = 'I(t)')
    plt.plot(t, u[:,4], label = 'Ia(t)')
    plt.plot(t, u[:,5], label = 'R(t)')
    plt.legend()
    plt.show()

u,t = solve_SEIR(T=100,dt=1.0,S_0=5e6,E2_0=100)
plot_SEIR(u,t)

"""
Run example:

user$ python3 seir_func.py

plots attached


"""
