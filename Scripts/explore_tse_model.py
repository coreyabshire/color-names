# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 20:10:28 2016

@author: corey
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def sigmoid(x, y, beta):
    return 1.0 / (1.0 + np.exp(-1 * beta * x))

beta = 4.0

x = np.arange(-5,5,0.02)
y = np.arange(-5,5,1.0)
x, y = np.meshgrid(x, y)
z = sigmoid(x, y, beta)

fig = plt.figure(figsize=(8.0,4.0))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, antialiased=False, shade=True)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0.0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(3))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.xaxis.set_label_text('x')
ax.yaxis.set_label_text('y')
ax.view_init(elev=40.0, azim=225.0)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('out.png', dpi=300)
plt.show()
