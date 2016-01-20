import random
from colornames.convex_hull_classifier import ConvexHullClassifier
from colornames.diagnostics import write_diagnostic_html, random_cie_colors, make_cie_gradient
import common

coords, names = common.read_old_data('../../data/old_data.csv')

thresholds = [2.0, 0.0, 0.0]
clf = ConvexHullClassifier(thresholds)
clf.fit(coords, names)

y, dist, thresh, ynames, inhull, numhulls = clf.predict(coords)
write_diagnostic_html('../color_page.html', coords, names, y, ynames, dist, thresh, inhull, numhulls)

random.seed(123456)
rcie = random_cie_colors(200)
yr, distr, threshr, ynamesr, inhullr, numhullsr = clf.predict(rcie)
write_diagnostic_html('../color_rcie.html', rcie, None, yr, ynamesr, distr, threshr, inhullr, numhullsr)

gcie = make_cie_gradient(200, clf.hull_centroid('RED'),
                         clf.hull_centroid('ORANGE'))
yg, distg, threshg, ynamesg, inhullg, numhullsg = clf.predict(gcie)
write_diagnostic_html('../color_gcie.html', gcie, None, yg, ynamesg, distg, threshg, inhullg, numhullsg)
