import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

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

    def __init__(self, thresholds):
        self.hulls = {}
        self.hullsd = {}
        self.thresholds = thresholds

    def fit(self, X, y):
        for yi in unique(y):
            print 'fitting', yi
            ccoords = coords[y==yi]
            hull = ConvexHull(ccoords)
            self.hulls[yi] = hull
            hulld = Delaunay(ccoords.irow(hull.vertices))
            self.hullsd[yi] = hulld

    def min_dist(self, xi):
        d = {}
        ks = self.hullsd.keys()
        for yi in self.hullsd.keys():
            h = self.hullsd[yi]
            d[yi] = min(abs(h.plane_distance(xi)))
        ss = sorted(ks, key=lambda k: d[k])
        sd = dict((k,0.0) for k in ks)
        sd[ss[0]] = d[ss[0]]
        ssum = d[ss[0]]
        th = 0
        if ssum == 0:
            sd[ss[0]] = 1.0
            ssum = 1.0
        else:
            while ss[th+1] < self.thresholds[th]:
                sd[ss[th+1]] = d[ss[th+1]]
                ssum += d[ss[th+1]]
                th += 1
        return pd.Series([sd[k]/ssum for k in ks], index=ks)

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

        dist = X.apply(self.min_dist, 1)

        ynames = dist.apply(lambda x: np.argmax(x), 1)
        
        return y, dist, ynames

def random_cie_colors(n):
    return pd.DataFrame({'cie_lstar': randn(n) * 10.0 + 60.0,
                         'cie_astar': randn(n) * 30,
                         'cie_bstar': randn(n) * 30},
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

def save_page(filename, coords, names, ynames):
    with open(filename, 'w') as outfile:
        outfile.write('<!doctype html>')
        outfile.write('<html>')
        outfile.write('<head>')
        outfile.write('</head>')
        outfile.write('<body>')
        outfile.write('<table>')
        outfile.write('<tr>')
        if names is not None:
	    outfile.write('<th>name</th>')
	outfile.write('<th>yname</th>')
	outfile.write('<th>L*</th>')
	outfile.write('<th>a*</th>')
	outfile.write('<th>b*</th>')
        outfile.write('</tr>')
        for i in range(len(ynames)):
            lab = LabColor(coords.iloc[i,0],coords.iloc[i,1],coords.iloc[i,2])
            rgb = convert_color(lab, sRGBColor, target_illuminant='d50')
            r = rgb.clamped_rgb_r
            g = rgb.clamped_rgb_g
            b = rgb.clamped_rgb_b
            h = sRGBColor(r,g,b).get_rgb_hex()
            outfile.write('<tr>')
            if names is not None:
                outfile.write('<td>%s</td>' % names.iloc[i])
            outfile.write('<td style="background: %s">%s</td>' % (h, ynames.iloc[i]))
            outfile.write('<td>%s</td>' % coords.iloc[i,0])
            outfile.write('<td>%s</td>' % coords.iloc[i,1])
            outfile.write('<td>%s</td>' % coords.iloc[i,2])
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



#ax.set_xlabel('CIE A*')
#ax.set_ylabel('CIE L*')
#ax.set_zlabel('CIE B*')

#plt.show()

thresholds = [0.01,0.1,1.0]
clf = ConvexHullFuzzyClassifier(thresholds)
clf.fit(coords, names)
y, dist, ynames = clf.predict(coords)
seed(123)
rcie = random_cie_colors(200)
yr, distr, ynamesr = clf.predict(rcie)

save_page('color_page.html', coords, names, ynames)
save_page('color_rcie.html', rcie, None, ynamesr)
