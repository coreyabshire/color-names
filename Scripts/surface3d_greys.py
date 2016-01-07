from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = '../../../../data/old_data.csv'
data = pd.read_csv(filename)
data = data.ix[0:len(data)-5,:]
coords = data.ix[:,2:5]
names = data.ix[:,5]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = coords.ix[:,0]
y = coords.ix[:,1]
z = coords.ix[:,2]
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

plt.show()
