import numpy as np
import pandas as pd
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


def point_lab(p):
    return LabColor(p[0],p[1],p[2])

def point_rgb(p):
    return convert_color(point_lab(p), sRGBColor)
    
def point_rgb255(p):
    rgb = point_rgb(p)
    return np.array([rgb.clamped_rgb_r * 255.0,
                     rgb.clamped_rgb_g * 255.0,
                     rgb.clamped_rgb_b * 255.0])

def random_cie_colors(n):
    return pd.DataFrame({'cie_lstar': np.round(np.random.randn(n) * 10.0 + 60.0, 2),
                         'cie_astar': np.round(np.random.randn(n) * 30, 2),
                         'cie_bstar': np.round(np.random.randn(n) * 30, 2)},
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

def write_diagnostic_html(filename, coords, names, y, ynames, dist, thresh, inhull, numhulls):
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
	outfile.write('<th>numhulls</th>')
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
	    outfile.write('<td>%s</td>' % numhulls.iloc[i])
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
