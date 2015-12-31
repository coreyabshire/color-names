import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

filename = '../data/old_data.csv'
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

class ConvexHullFuzzyClassifier(object):

    def __init__(self):
        self.hulls = {}
        self.hullsd = {}

    def fit(self, X, y):
        for yi in unique(y):
            print 'fitting', yi
            ccoords = coords[y==yi]
            hull = ConvexHull(ccoords)
            self.hulls[yi] = hull
            hulld = Delaunay(ccoords.irow(hull.vertices))
            self.hullsd[yi] = hulld

    def predict(self, X):
        y = {}
        np.ndarray((len(X),len(self.hullsd.keys())))
        for yi in self.hullsd.keys():
             y[yi] = self.hullsd[yi].find_simplex(X)
        y = pd.DataFrame(y)

        # The output of find_simplex above will return the index of the
        # simplex for the point given and -1 otherwise, but for our function
        # we need the fuzzy values. To get there, we first convert to a simple
        # 0 and 1 based on those results.
        y = y.apply(lambda col: col.apply(lambda v: 0 if v < 0 else 1))

        # If a point falls within two hulls, we have to normalize so that
        # the value for all classes sum to one.
        colsum = sum(y, 1) # compute the current col sums
        y = y.apply(lambda col: col / colsum)
        
        return y

def random_cie_colors(n):
    return pd.DataFrame({'cie_lstar': randn(n) * 10 + 50,
                         'cie_astar': randn(n) * 30,
                         'cie_bstar': randn(n) * 30})
    
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
 

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#show_hull('NEUTRAL', 'grey')
#show_hull('RED', 'red')
#show_hull('YELLOW', 'yellow')
#show_hull('ORANGE', 'orange')
#show_hull('BROWN', 'brown')
#show_hull('PURPLEVIOLET', 'purple')
#show_hull('MAGENTA', 'magenta')
#show_hull('GREEN', 'green')
#show_hull('BLUECYAN', 'blue')



#ax.set_xlabel('CIE A*')
#ax.set_ylabel('CIE L*')
#ax.set_zlabel('CIE B*')

#plt.show()

clf = ConvexHullFuzzyClassifier()
clf.fit(coords, names)
y = clf.predict(coords)
