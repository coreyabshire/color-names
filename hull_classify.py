import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from pointTriangleDistance import pointTriangleDistance


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

class ConvexHullFuzzyClassifier(object):

    def __init__(self, thresholds):
        self.ks = []
        self.hulls = {}
        self.thresholds = thresholds

    def fit(self, X, y):
        self.ks = unique(y)
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
        colsum = sum(y, 1) # compute the current col sums
        inhull = colsum > 0
        y[inhull] = y[inhull].apply(lambda c: c / colsum)

        # For points outside any hulls, we need a distance matrix.
        dist = X.apply(self.distance_vector, 1)
        thresh = dist.apply(self.vector_threshold, 1)
        for k in y:
            y.loc[~inhull,k] = thresh[~inhull][k]

        ynames = y.apply(lambda x: np.argmax(x), 1)
        
        return y, dist, thresh, ynames, inhull

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

def point_lab(p):
    return LabColor(p[0],p[1],p[2])

def point_rgb(p):
    return convert_color(point_lab(p), sRGBColor)
    
def point_rgb255(p):
    rgb = point_rgb(p)
    return np.array([rgb.clamped_rgb_r * 255.0,
                     rgb.clamped_rgb_g * 255.0,
                     rgb.clamped_rgb_b * 255.0])
def sigmix(a, b):
    d = (a + b)
    beta = 1.0 / (d / 16.0)
    x = -(d / 2.0) + a
    return 1.0 - (1.0 / (1.0 + np.exp(-beta * x)))

def random_cie_colors(n):
    return pd.DataFrame({'cie_lstar': np.round(randn(n) * 10.0 + 60.0, 2),
                         'cie_astar': np.round(randn(n) * 30, 2),
                         'cie_bstar': np.round(randn(n) * 30, 2)},
                        columns=['cie_lstar','cie_astar','cie_bstar'])



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

def save_page(filename, coords, names, y, ynames, dist, thresh, inhull):
    with open(filename, 'w') as outfile:
        outfile.write('<!doctype html>')
        outfile.write('<html>')
        outfile.write('<head>')
        outfile.write('<link type="text/css" rel="stylesheet" href="color_page.css">')
        outfile.write('</head>')
        outfile.write('<body>')
        outfile.write('<table>')
        outfile.write('<tr>')
	outfile.write('<th>patch</th>')
        if names is not None:
	    outfile.write('<th>name</th>')
	outfile.write('<th>yname</th>')
	outfile.write('<th>L*</th>')
	outfile.write('<th>a*</th>')
	outfile.write('<th>b*</th>')
	outfile.write('<th>r</th>')
	outfile.write('<th>g</th>')
	outfile.write('<th>b</th>')
	outfile.write('<th>inhull</th>')
        for k in y:
	    outfile.write('<th>Y-%s</th>' % k[:2])
        for k in dist:
	    outfile.write('<th>D-%s</th>' % k[:2])
        for k in thresh:
	    outfile.write('<th>T-%s</th>' % k[:2])
        outfile.write('</tr>')
        for i in range(len(ynames)):
            lab = LabColor(coords.iloc[i,0],coords.iloc[i,1],coords.iloc[i,2])
            rgb = convert_color(lab, sRGBColor, target_illuminant='d50')
            r = rgb.clamped_rgb_r
            g = rgb.clamped_rgb_g
            b = rgb.clamped_rgb_b
            h = sRGBColor(r,g,b).get_rgb_hex()
            outfile.write('<tr>')
            outfile.write('<td style="background: %s"></td>' % h)
            if names is not None:
                outfile.write('<td>%s</td>' % names.iloc[i])
            outfile.write('<td>%s</td>' % ynames.iloc[i])
            outfile.write('<td class="num">%.2f</td>' % coords.iloc[i,0])
            outfile.write('<td class="num">%.2f</td>' % coords.iloc[i,1])
            outfile.write('<td class="num">%.2f</td>' % coords.iloc[i,2])
            outfile.write('<td class="num">%.2f</td>' % r)
            outfile.write('<td class="num">%.2f</td>' % g)
            outfile.write('<td class="num">%.2f</td>' % b)
	    outfile.write('<td>%s</td>' % inhull.iloc[i])
            for k in y:
	        outfile.write('<td class="num">%.2f</td>' % y.iloc[i][k])
            for k in dist:
	        outfile.write('<td class="num">%.2f</td>' % dist.iloc[i][k])
            for k in thresh:
                t = thresh.iloc[i][k]
                if t > 9999999.00:
	            outfile.write('<td class="num">%.2f</td>' % t)
                else:
	            outfile.write('<td class="num">-</td>')
            outfile.write('</tr>')
        outfile.write('</table>')
        outfile.write('</body>')
        outfile.write('</html>')

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

def make_cie_gradient(n, a, b):
    x = np.linspace(1.0, 0.0, n)
    return pd.DataFrame({'cie_lstar': a[0] * x + b[0] * (1.0-x),
                         'cie_astar': a[1] * x + b[1] * (1.0-x),
                         'cie_bstar': a[2] * x + b[2] * (1.0-x)},
                        columns=['cie_lstar','cie_astar','cie_bstar'])

#ax.set_xlabel('CIE A*')
#ax.set_ylabel('CIE L*')
#ax.set_zlabel('CIE B*')

#plt.show()

thresholds = [2.0,0.0,0.0]
clf = ConvexHullFuzzyClassifier(thresholds)
clf.fit(coords, names)

y, dist, thresh, ynames, inhull = clf.predict(coords)
save_page('color_page.html', coords, names, y, ynames, dist, thresh, inhull)

seed(123456)
rcie = random_cie_colors(200)
yr, distr, threshr, ynamesr, inhullr = clf.predict(rcie)
save_page('color_rcie.html', rcie, None, yr, ynamesr, distr, threshr, inhullr)

gcie = make_cie_gradient(200, clf.hull_centroid('RED'),
                         clf.hull_centroid('ORANGE'))
yg, distg, threshg, ynamesg, inhullg = clf.predict(gcie)
save_page('color_gcie.html', gcie, None, yg, ynamesg, distg, threshg, inhullg)
