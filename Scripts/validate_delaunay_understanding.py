import numpy as np
from scipy.spatial import Delaunay, ConvexHull

from colornames import pointTriangleDistance

ps = np.array([[ 0.0, -1.0,  1.0],
               [ 0.0,  1.0,  1.0],
               [ 0.0,  1.0, -1.0],
               [ 0.0, -1.0, -1.0],
               [-1.0, -1.0,  1.0],
               [-1.0,  1.0,  1.0],
               [-1.0,  1.0, -1.0],
               [-1.0, -1.0, -1.0]])
d = Delaunay(ps)
h = ConvexHull(ps)

print 'Delaunay'
for s in d.simplices:
    print s
    
print 'Convex hull'
for s in h.simplices:
    tri = np.array([[x for x in h.points[s[i]]] for i in range(3)])
    print tri
    p = np.array([2,0,0])
    print p
    print pointTriangleDistance(tri, p)
