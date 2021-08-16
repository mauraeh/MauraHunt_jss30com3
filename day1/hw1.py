import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {'decision_variables': ['mug', 'bowl'],'time': [4, 3], 'clay': [1, 1.5], 'profit': [2,3]}

data = pd.DataFrame(data=data)

constraints = {'time': [1200], 'clay': [450]}
constraints = pd.DataFrame(data=constraints)

#objective maximize profit: let m = mugs and b = bowls
mug = data[data.decision_variables == 'mug']
bowl = data[data.decision_variables == 'bowl']
b = np.array(range(0,300))

obj = -bowl.profit[1]/mug.profit[0] * b
con1 = constraints.time[0]/mug.time[0] - bowl.time[1]/mug.time[0] * b
con2 = constraints.clay[0]-bowl.clay[1] * b

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

solution = line_intersection(([b[0], con1[0]], [b[len(b)-1], con1[len(b)-1]]), 
                        ([b[0], con2[0]], [b[len(b)-1], con2[len(b)-1]]))



plt.plot(b, con1,linestyle='dashed')
plt.plot(b, con2,linestyle='dashed')
plt.scatter(x=solution[0],y=solution[1])
plt.xlabel('Bowls')
plt.ylabel('Mugs')
plt.show()
#plt.savefig('graphical_results.png')


import pyomo.environ as pyo
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)

model.OBJ = pyo.Objective(expr =  mug.profit[0]*model.x[1] + bowl.profit[1]*model.x[2])

model.Constraint1 = pyo.Constraint(expr = mug.time[0]*model.x[1] + bowl.time[1]*model.x[2] <= constraints.time[0])
model.Constraint2 = pyo.Constraint(expr = mug.clay[0]*model.x[1] + bowl.clay[1]*model.x[2] <= constraints.clay[0])

opt = pyo.SolverFactory('glpk')
# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance)
instance.display()