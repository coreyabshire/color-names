import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from pointTriangleDistance import pointTriangleDistance

def sigmix(a, b):
    d = (a + b)
    beta = 1.0 / (d / 16.0)
    x = -(d / 2.0) + a
    return 1.0 - (1.0 / (1.0 + np.exp(-beta * x)))

class Hull(object):

    def __init__(self, points):
        self.hull = ConvexHull(points)
        self.delaunay = Delaunay(points.irow(self.hull.vertices))
        self.tris = [np.array([[x for x in self.hull.points[s[i]]] for i in range(3)]) for s in self.hull.simplices]

    def centroid(self):
        return self.delaunay.points.mean(0)

    def distance(self, xi):
        ds = [pointTriangleDistance(tri, xi)[0] for tri in self.tris]
        #return min(np.abs(self.delaunay.plane_distance(xi)))
        return min(ds)

    def contains(self, X):
        return self.delaunay.find_simplex(X)

class ConvexHullClassifier(object):

    def __init__(self, thresholds):
        self.ks = []
        self.hulls = {}
        self.thresholds = thresholds
        print 'initialized'

    def fit(self, X, y):
        self.ks = np.unique(y)
        self.hulls = {k:Hull(X[y==k]) for k in self.ks}

    def predict(self, X):
        y = pd.DataFrame({k:self.hulls[k].contains(X) for k in self.ks})

        # The output of find_simplex above will return the index of the
        # simplex for the point given and -1 otherwise, but for our function
        # we need the fuzzy values. To get there, we first convert to a simple
        # 0 and 1 based on those results.
        y = y.apply(lambda c: c.apply(lambda v: 0 if v < 0 else 1))

        # If a point falls within two hulls, we have to normalize so that
        # the value for all classes sum to one.
        colsum = np.sum(y, 1) # compute the current col sums
        inhull = colsum > 0
        numhulls = colsum
        y[inhull] = y[inhull].apply(lambda c: c / colsum)

        # For points outside any hulls, we need a distance matrix.
        dist = X.apply(self.distance_vector, 1)
        thresh = dist.apply(self.vector_threshold, 1)
        for k in y:
            y.loc[~inhull,k] = thresh[~inhull][k]

        ynames = y.apply(lambda x: np.argmax(x), 1)
        
        return y, dist, thresh, ynames, inhull, numhulls

    def distance_vector(self, xi):
        """Finds the minimum distance of each point in xi to each hull."""
        d = {k:self.hulls[k].distance(xi) for k in self.ks}
        return pd.Series([d[k] for k in self.ks], index=self.ks)

    def vector_threshold(self, d):
        # We then apply the threshold function so that only up to 
        # four categories are considered.
        ss = sorted(self.ks, key=lambda k: d[k])
        sd = {k:0.0 for k in self.ks}
        sd[ss[0]] = d[ss[0]]
        ssum = d[ss[0]]
        th = 0
        inthresh = [ss[0]]
        if ssum == 0:
            sd[ss[0]] = 1.0
            ssum = 1.0
        else:
            for t in range(len(self.thresholds)):
                j = ss[t+1] # color key to test at this t
                if d[j] < self.thresholds[t]:
                    inthresh.append(j)
                    sd[j] = d[j]
                    ssum += d[j]
                    th += 1
                else:
                    break
        # Then for each pair of distances we apply the sigmoid function
        # to adjust the transition between hulls, and then apply weights.
        if len(inthresh) == 1:
            sd[inthresh[0]] = 1.0
        elif len(inthresh) == 2:
            sd[inthresh[0]] = sigmix(sd[inthresh[0]], sd[inthresh[1]])
            sd[inthresh[1]] = 1.0 - sd[inthresh[0]]
        elif len(inthresh) == 3:
            sd[inthresh[0]] = sigmix(sd[inthresh[0]], sd[inthresh[1]])
            sd[inthresh[1]] = 1.0 - sd[inthresh[0]]
        else:
            for i,j in [(i,i+1) for i in range(len(inthresh)-1)]:
                a = sd[inthresh[i]]
                b = sd[inthresh[j]]
                total = a + b
        return pd.Series([sd[k] for k in self.ks], index=self.ks)

    def hull_centroid(self, k):
        return self.hulls[k].centroid()
