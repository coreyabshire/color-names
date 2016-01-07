import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colornames.convex_hull_classifier import ConvexHullClassifier
from colornames.diagnostics import write_diagnostic_html, random_cie_colors, make_cie_gradient

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


#ax.set_xlabel('CIE A*')
#ax.set_ylabel('CIE L*')
#ax.set_zlabel('CIE B*')

#plt.show()

thresholds = [2.0,0.0,0.0]
clf = ConvexHullClassifier(thresholds)
clf.fit(coords, names)

y, dist, thresh, ynames, inhull = clf.predict(coords)
write_diagnostic_html('color_page.html', coords, names, y, ynames, dist, thresh, inhull)

random.seed(123456)
rcie = random_cie_colors(200)
yr, distr, threshr, ynamesr, inhullr = clf.predict(rcie)
write_diagnostic_html('color_rcie.html', rcie, None, yr, ynamesr, distr, threshr, inhullr)

gcie = make_cie_gradient(200, clf.hull_centroid('RED'),
                         clf.hull_centroid('ORANGE'))
yg, distg, threshg, ynamesg, inhullg = clf.predict(gcie)
write_diagnostic_html('color_gcie.html', gcie, None, yg, ynamesg, distg, threshg, inhullg)
