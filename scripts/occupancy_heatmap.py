from descartes import PolygonPatch
import geotiler
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Script to make the occupancy distribution of taxis over the Manhattan
# neighborhoods.  It uses geotiler to draw a map of the area.  Note that
# geotiler requires Python 3.
#
# USAGE:
#         python3 occupancy_heatmap.py

def GetPolysNbrhoods(geojson_file):
    ''' Get the polygon for each neighborhood. '''
    with open(geojson_file) as f:
        js = json.load(f)

    polys = [feature['geometry'] for feature in js['features']]
    nbrhoods = [feature['properties']['neighborhood'] for feature in js['features']]
    return polys, nbrhoods

def GetBBox(polys):
    ''' Compute bounding box for the polygons. '''
    gxmin = None
    gymin = None
    gxmax = None
    gymax = None
    
    for poly, nbrhood in zip(polys, nbrhoods):
        coords = poly['coordinates'][0]
        cxmin = min([xy[0] for xy in coords])
        cymin = min([xy[1] for xy in coords])
        cxmax = max([xy[0] for xy in coords])
        cymax = max([xy[1] for xy in coords])
        if gxmin == None or cxmin < gxmin: gxmin = cxmin
        if gymin == None or cymin < gymin: gymin = cymin
        if gxmax == None or cxmax > gxmax: gxmax = cxmax
        if gymax == None or cymax > gymax: gymax = cymax

    return [gxmin, gymin, gxmax, gymax]

def GetSeqs(seq_file):
    ''' Read the trajectory sequences from seq_file. '''
    seqs = []
    with open(seq_file) as f:
        for line in f:
            seqs.append([int(x) for x in line.strip().split(',')])
    return seqs

def ReadKeys(key_file):
    ''' Read the neighborhood keys from key_file. '''
    with open(key_file, 'rb') as f:
        nk = pickle.load(f)
        del nk[0]
        nk['OTHER'] = 0
        return nk

def AlphaMap(seqs, nbrhood_keys):
    max_ind = np.max(np.max(seqs))
    vec = np.zeros(max_ind + 1)
    for seq in seqs:
        for loc in seq:
            vec[loc] += 1
    vec /= np.sum(vec)
    return {key:vec[val] for key, val in nbrhood_keys.items()}

if __name__ == '__main__':
    polys, nbrhoods = GetPolysNbrhoods('../data/neighborhoods_Manhattan.geojson')
    nbrhood_keys = ReadKeys('../processed_data/taxi/neighborhood_keys.pkl')
    seqs = GetSeqs('../processed_data/taxi/manhattan-year-seqs.txt')
    bbox = GetBBox(polys)
    alpha_map = AlphaMap(seqs, nbrhood_keys)

    # Display the image
    fig = plt.figure()
    ax = fig.gca()

    toner = geotiler.find_provider('stamen-toner-lite')
    mm = geotiler.Map(extent=bbox, zoom=12, provider=toner)
    img = geotiler.render_map(mm)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for poly, nbrhood in zip(polys, nbrhoods):
        coords = poly['coordinates'][0]
        poly['coordinates'][0] = [mm.rev_geocode(p) for p in coords]
        patch = PolygonPatch(poly, alpha=-1.0 / np.log(alpha_map[nbrhood]))
        ax.add_patch(patch)

    # Increase the dpi to save better quality figures.  For some reason,
    # matplotlib cannot save the rendered map as a vectorized image.
    plt.savefig('taxi-distribution.png', dpi=200, bbox_inches='tight')
    plt.show()
