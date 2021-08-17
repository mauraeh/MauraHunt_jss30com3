import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
import matplotlib
from matplotlib import pyplot
import random
from numpy import exp

def penalty(con):
    return "fix me"


def objective(x):
    return np.abs(x[0]) + 0.3 * np.sin(15*x[0]) + abs(x[1]) + 0.3 * np.sin(15 * x[1]) 

# black-box optimization software
def SA2D(objective, bounds, n_iterations, step_size, temp, init):
  # generate an initial point
  best = init
  # evaluate the initial point
  best_eval = objective(best)
  curr, curr_eval = best, best_eval # current working solution
  scores = list()
  points=list()
  c = list()
  cscore = list()
  for i in range(n_iterations): # take a step
    # take a step
    candidate = [curr[0] +rand()*step_size[0]-step_size[0]/2.0,
                 curr[1]+rand()*step_size[1]-step_size[1]/2.0]

    for j in range(len(candidate)):
        if candidate[j] < bounds['lb']:
            candidate[j] = bounds['lb']
        if candidate[j] > bounds['ub']:
            candidate[j] = bounds['ub']

    points.append(candidate)
    # evaluate candidate point
    candidate_eval = objective(candidate)
    # keep track of scores scores
    scores.append(candidate_eval)
    if (candidate_eval < best_eval): # store new best point
      best, best_eval = candidate, candidate_eval
      #report progress
      print('>%d f(%s) = %.5f, %s' % (i, best, best_eval,candidate))	
    # difference between candidate and current point evaluation
    diff = candidate_eval - curr_eval
    # calculate temperature for current epoch
    for e in range(len(temp)):
        t = (temp[e]/float(i+1))
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if (diff < 0) or (rand() < metropolis):
          # store the new current point
          curr, curr_eval = candidate, candidate_eval
    c.append(curr)
    cscore.append(curr_eval)
  return (best, best_eval, points, scores, c, cscore)
# Set random seed
random.seed(1)
bounds=asarray([[-3.0,3.0],[-3.0,3.0]])
bounds = {
  "lb": -3.0,
  "ub": 3.0,
}
step_size=[0.8,0.8]
n_iterations=100
init=[2.4,2.0]
temp = [37.5]
best, score, points, scores,c,cscore = SA2D(objective, bounds, n_iterations, step_size, temp, init)
n, m = 7, 7
start = -3
x_vals = np.arange(start, start+7, 0.1)
y_vals = np.arange(start, start+7, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)
print(X)
print(Y)
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
Z = np.abs(X) + 0.3 * np.sin(15*X) + abs(Y) + 0.3 * np.sin(15 * Y)
cp = ax.contour(X, Y, Z)
ax.clabel(cp, inline=True, fontsize=10)
ax.set_title('Contour Plot')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
for i in range(n_iterations):
    plt.plot(points[i][0],points[i][1],"o");
plt.show()