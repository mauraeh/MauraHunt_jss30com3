import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)

model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])

model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)


import numpy as np

import matplotlib.pyplot as plt
from desdeo_problem.problem import MOProblem
from desdeo_problem.problem import variable_builder
from desdeo_problem.problem import _ScalarObjective

def f_1(xs: np.ndarray):
    xs = np.atleast_2d(xs)
    xs_plusone = np.roll(xs, 1, axis=1)
    return np.sum(-10*np.exp(-0.2*np.sqrt(xs[:, :-1]**2 + xs_plusone[:, :-1]**2)), axis=1)

def f_2(xs: np.ndarray):
    xs = np.atleast_2d(xs)
    return np.sum(np.abs(xs)**0.8 + 5*np.sin(xs**3), axis=1)


varsl = variable_builder(
    ["x_1", "x_2", "x_3"],
    initial_values=[0, 0, 0],
    lower_bounds=[-5, -5, -5],
    upper_bounds=[5, 5, 5],
)

f1 = _ScalarObjective(name="f1", evaluator=f_1)
f2 = _ScalarObjective(name="f2", evaluator=f_2)

problem = MOProblem(variables=varsl, objectives=[f1, f2], ideal=np.array([-20, -12]), nadir=np.array([-14, 0.5]))

from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation

p_front = solve_pareto_front_representation(problem, step=1.0)[1]

plt.scatter(p_front[:, 0], p_front[:, 1], label="Pareto front")
plt.scatter(problem.ideal[0], problem.ideal[1], label="Ideal")
plt.scatter(problem.nadir[0], problem.nadir[1], label="Nadir")
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("Approximate Pareto front of the Kursawe function")
plt.legend()
plt.show()