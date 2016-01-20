import pandas as pd
from colornames.convex_hull_classifier import ConvexHullClassifier
from colornames.diagnostics import write_diagnostic_html
from common import read_old_data

coords, names = read_old_data('../../data/old_data.csv')

thresholds = [2.0, 0.0, 0.0]
clf = ConvexHullClassifier(thresholds)
clf.fit(coords, names)

new_data_filename = '../../data/new_data.csv'
new_data = pd.read_csv(new_data_filename)
new_coords = new_data.ix[:, 2:5]

y, dist, thresh, ynames, inhull, numhulls = clf.predict(new_coords)
write_diagnostic_html('../color_new.html', new_coords, None, y, ynames, dist, thresh, inhull, numhulls)

dist_sort = dist.values
dist_sort.sort(axis=1)
dist_sort = pd.DataFrame(dist_sort, dist.index)
dist_sort.columns = ['dist_sort_%s' % c for c in dist_sort.columns]

rank = dist_sort.iloc[:, 0].copy()
rank.columns = ['rank']
rank[numhulls == 0] = 0
rank[numhulls >= 2] = 0

subset = pd.concat([new_data, rank, inhull, numhulls, dist_sort.iloc[:, 0:3]], axis=1)
subset.columns = ['line_num', 'color_patch', 'cie_lstar', 'cie_astar', 'cie_bstar', 'rank', 'inhull', 'numhulls', 'dist0', 'dist1', 'dist2']
subset = subset.sort('rank')
subset.to_csv('../subset.csv', index=False)
