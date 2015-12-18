import json
from shapely.geometry import shape, Point
import pickle
import numpy as np

def GetPolysNbrhoods(geojson_file):
    with open(geojson_file) as f:
        js = json.load(f)

    polys = [feature['geometry'] for feature in js['features']]
    nbrhoods = [feature['properties']['neighborhood'] for feature in js['features']]
    return polys, nbrhoods

def GetBBox(polys, nbrhoods):
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

    xdiff = gxmax - gxmin
    ydiff = gymax - gymin
    f = 0
    return [gxmin + f * xdiff, gymin + f * ydiff,
            gxmax - f * xdiff, gymax - f * ydiff]

def GetSeqs(seq_file):
    seqs = []
    with open(seq_file) as f:
        for line in f:
            seqs.append([int(x) for x in line.strip().split(',')])
    return seqs

def ReadKeys(key_file):
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

