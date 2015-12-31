import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

filename = '../../../../data/old_data.csv'
data = pd.read_csv(filename)
data = data.ix[0:len(data)-5,:]
coords = data.ix[:,2:5]
names = data.ix[:,5]

namemap = {
    'YELLOW': 'y',
    'MAGENTA': 'm', 
    'BLUECYAN': 'c', 
    'NEUTRAL': 'k', 
    'ORANGE': 'r', 
    'GREEN': 'g',
    'BROWN': 'r', 
    'RED': 'r', 
    'PURPLEVIOLET': 'b'}

c = names.apply(lambda x: namemap[x])

coords = coords[names=='RED']

hull = ConvexHull(coords)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = coords.ix[:,1]
ys = coords.ix[:,0]
zs = coords.ix[:,2]
ax.scatter(xs, ys, zs, c='red', marker='o')

for simplex in hull.simplices:
    s = coords.irow(simplex)
    print s
    sx = list(s.ix[:,1])
    sy = list(s.ix[:,0])
    sz = list(s.ix[:,2])
    sx.append(sx[0])
    sy.append(sy[0])
    sz.append(sz[0])
    ax.plot(sx, sy, sz, 'r-')


ax.set_xlabel('CIE A*')
ax.set_ylabel('CIE L*')
ax.set_zlabel('CIE B*')

plt.show()
