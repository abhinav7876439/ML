from sympy import symbols, Eq, nsolve
import numpy as np
import matplotlib.pyplot as plt

# Define variables
pi00, pi01, pi02, pi10, pi11, pi12 = symbols('pi00 pi01 pi02 pi10 pi11 pi12')
λ, θ, μ, C = symbols('λ θ μ C')

# Substitute specific values
params = {
    λ: 0.5,   # demand rate
    θ: 0.2,   # perish rate
    μ: 0.7,   # service rate
    C: 1.0    # switch rate
}

# Define equations
eq1 = Eq((params[λ] + params[θ])*pi01 + params[C]*pi10 - params[C]*pi00, 0)
eq2 = Eq((params[λ] + 2*params[θ])*pi02 + params[C]*pi11 - (params[λ] + params[θ])*pi01 - params[C]*pi01, 0)
eq3 = Eq(params[C]*pi12 - (params[λ] + 2*params[θ])*pi02 - params[C]*pi02, 0)
eq4 = Eq((params[λ] + params[θ])*pi11 + params[C]*pi00 - params[μ]*pi10 - params[C]*pi10, 0)
eq5 = Eq(params[μ]*pi10 + (params[λ] + 2*params[θ])*pi12 + params[C]*pi01 - (params[λ] + params[θ])*pi11 - params[μ]*pi11 - params[C]*pi11, 0)
eq6 = Eq(params[μ]*pi11 + params[C]*pi02 - (params[λ] + 2*params[θ])*pi12 - params[C]*pi12, 0)
eq7 = Eq(pi00 + pi01 + pi02 + pi10 + pi11 + pi12, 1)

from sympy import Matrix

# Initial guess (all probabilities positive and sum to 1)
guess = Matrix([0.2, 0.2, 0.1, 0.2, 0.2, 0.1])

try:
    sol = nsolve([eq1, eq2, eq3, eq4, eq5, eq6, eq7], (pi00, pi01, pi02, pi10, pi11, pi12), guess)
    steady_state = {
        pi00: float(sol[0]),
        pi01: float(sol[1]),
        pi02: float(sol[2]),
        pi10: float(sol[3]),
        pi11: float(sol[4]),
        pi12: float(sol[5]),
    }
    print("Steady-state probabilities:")
    for k, v in steady_state.items():
        print(f"{k}: {v:.4f}")

    # Bar plot
    labels = [str(k) for k in steady_state.keys()]
    values = [v for v in steady_state.values()]
    plt.bar(labels, values)
    plt.ylabel("Probability")
    plt.title("Steady-State Distribution")
    plt.show()
except Exception as e:
    print("No solution found.", e)
