import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay

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


def show_hull(cname, ccol):
    ccoords = coords[names==cname]

    hull = ConvexHull(ccoords)

    xs = ccoords.ix[:,0]
    ys = ccoords.ix[:,1]
    zs = ccoords.ix[:,2]
    #ax.scatter(xs, ys, zs, c=ccol, marker='o')

    for simplex in hull.simplices:
        s = ccoords.irow(simplex)
        #print s
        sx = list(s.ix[:,0])
        sy = list(s.ix[:,1])
        sz = list(s.ix[:,2])
        sx.append(sx[0])
        sy.append(sy[0])
        sz.append(sz[0])
        ax.plot(sx, sy, sz, ccol, alpha=0.2)

    hulld = Delaunay(ccoords.irow(hull.vertices))
    hulld.find_simplex(coords)
    hcol = ['grey' if x<0 else 'green' for x in hulld.find_simplex(coords)]
    hxs = coords.ix[:,0]
    hys = coords.ix[:,1]
    hzs = coords.ix[:,2]
    ax.scatter(hxs, hys, hzs, c=hcol, marker='o', alpha=0.2)
 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#show_hull('NEUTRAL', 'grey')
#show_hull('RED', 'red')
#show_hull('YELLOW', 'yellow')
#show_hull('ORANGE', 'orange')
#show_hull('BROWN', 'brown')
#show_hull('PURPLEVIOLET', 'purple')
#show_hull('MAGENTA', 'magenta')
show_hull('GREEN', 'green')
#show_hull('BLUECYAN', 'blue')



ax.set_xlabel('CIE A*')
ax.set_ylabel('CIE L*')
ax.set_zlabel('CIE B*')

plt.show()
