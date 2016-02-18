import os
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from colornames.convex_hull_classifier import ConvexHull


def unique_col(x, allow_outlier=False):
    """Return the unique color chosen if it is indeed unique, otherwise nan."""
    if allow_outlier:
        v = x.value_counts().sort_values(0, False)
        if len(v) == 1:
            # all observers agree
            #print v
            return v.index[0]
        elif len(v) == 2 and v[1] == 1:
            # one observer disagrees (i.e., one outlier)
            #print v
            return v.index[0]
        else:
            # more than one outlier (or no data)
            return np.nan
    else:
        c = x.dropna().unique()
        if len(c) == 1:
            return c[0]
        else:
            return np.nan


def show_hull(names, coords, cname, ccol):
    ccoords = coords.ix[names[names == cname].index]
    print cname, len(ccoords)
    if len(ccoords) == 0:
        sys.stderr.write('empty coords set: %s\n' % cname)
        return

    hull = ConvexHull(ccoords)

    xs = ccoords.ix[:,1]
    ys = ccoords.ix[:,0]
    zs = ccoords.ix[:,2]
    #ax.scatter(xs, ys, zs, c=ccol, marker='o')

    for simplex in hull.simplices:
        s = ccoords.iloc[simplex]
        #print s
        sx = list(s.ix[:,1])
        sy = list(s.ix[:,0])
        sz = list(s.ix[:,2])
        sx.append(sx[0])
        sy.append(sy[0])
        sz.append(sz[0])
        ax.plot(sx, sy, sz, ccol, alpha=0.2)


DATA_DIR = 'C:/Users/Corey/Dropbox/Learning/Color Names/data'

coords = pd.read_csv(os.path.join(DATA_DIR, 'new_data.csv'))
coords = coords[['color_patch', 'cie_lstar', 'cie_astar', 'cie_bstar']]
coords.columns = ['patch', 'l', 'a', 'b']
coords.index = coords.patch
coords = coords[['l', 'a', 'b']]

obs = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
obs.index = obs.patch
del obs['patch']




#data = data.ix[0:len(data)-5,:]
#coords = data.ix[:,2:5]
names = obs.apply(lambda x: unique_col(x, True), 1).dropna()

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

#c = names.apply(lambda x: namemap[x])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
show_hull(names, coords, 'NEUTRAL', 'grey')
show_hull(names, coords, 'RED', 'red')
show_hull(names, coords, 'YELLOW', 'yellow')
show_hull(names, coords, 'ORANGE', 'orange')
show_hull(names, coords, 'BROWN', 'brown')
show_hull(names, coords, 'PURPLEVIOLET', 'purple')
show_hull(names, coords, 'MAGENTA', 'magenta')
show_hull(names, coords, 'GREEN', 'green')
show_hull(names, coords, 'BLUECYAN', 'blue')

ax.set_xlabel('CIE A*')
ax.set_ylabel('CIE L*')
ax.set_zlabel('CIE B*')

plt.show()
