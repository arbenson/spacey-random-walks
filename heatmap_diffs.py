import matplotlib.pyplot as plt
from collections import defaultdict
from descartes import PolygonPatch
import geotiler
from heatmaps_common import *

def ValueAlphaMap(vec, nbrhood_keys):
    vec /= np.sum(vec)
    return {key:vec[val] for key, val in nbrhood_keys.items()}

def ReadEntries(filename):
    with open(filename) as f:
        vals = defaultdict(list)
        for line in f:
            # PSRS(i, j, k) PSOMC(i, j, k) i j k
            data = line.strip().split()
            srw = float(data[0])
            somc = float(data[1])
            j = int(data[3])
            k = int(data[4])
            vals[(j, k)].append((srw, somc))
        
        srw_probs = {}
        somc_probs = {}
        for key, val in vals.items():
            p1, p2 = zip(*val)
            p1 = np.array(p1)
            p2 = np.array(p2)
            srw_probs[key] = p1
            somc_probs[key] = p2

        return srw_probs, somc_probs

if __name__ == '__main__':
    polys, nbrhoods = GetPolysNbrhoods('data/neighborhoods_Manhattan.geojson')
    nbrhood_keys = ReadKeys('processed_data/neighborhood_keys.pkl')
    print(nbrhood_keys)
    rev_keys = {val:key for key, val in nbrhood_keys.items()}
    bbox = GetBBox(polys, nbrhoods)
    #srw_probs, somc_probs = ReadEntries('processed_data/large_diffs.txt')
    srw_probs, somc_probs = ReadEntries('results.txt')

    z = 12
    toner = geotiler.find_provider('stamen-toner-lite')
    mm = geotiler.Map(extent=bbox, zoom=z, provider=toner)
    img = geotiler.render_map(mm)

    for poly, nbrhood in zip(polys, nbrhoods):
        coords = poly['coordinates'][0]
        poly['coordinates'][0] = [mm.rev_geocode(p) for p in coords]

    for key in srw_probs:
        print(rev_keys[key[1]], ' -> ', rev_keys[key[0]])

        alpha_map1 = ValueAlphaMap(srw_probs[key], nbrhood_keys)
        alpha_map2 = ValueAlphaMap(somc_probs[key], nbrhood_keys)

        for alpha_map in [alpha_map1, alpha_map2]:
            fig = plt.figure()
            ax = fig.gca()
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            for poly, nbrhood in zip(polys, nbrhoods):
                if alpha_map[nbrhood] > 0:
                    alpha = -1.0 / np.log(alpha_map[nbrhood])
                else:
                    alpha = 0.0
                patch = PolygonPatch(poly, alpha=alpha)
                ax.add_patch(patch)

            plt.show()
