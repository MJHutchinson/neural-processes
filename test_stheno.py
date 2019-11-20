from stheno import GP, EQ, Delta

import numpy as np

import matplotlib.pyplot as plt


gp = GP(EQ().stretch(0.8) * 0.5)

x = np.array([1.,2.,3.,4.,5.,6.])
y = np.array([4.,2.,6.,2.,6.,2.])

gp_cond = gp | (x, y)

x_lin = np.linspace(-4,11,200)

gp_cond_x = gp_cond(x_lin)

plt.figure()

mean, lower, upper = gp_cond_x.marginals()
plt.plot(x_lin, mean, c='g')
plt.plot(x_lin, lower, ls='--', c='g')
plt.plot(x_lin, upper, ls='--', c='g')

for i in range(3):
    plt.plot(x_lin, gp_cond_x.sample())

plt.scatter(x, y, c='b')


plt.show()