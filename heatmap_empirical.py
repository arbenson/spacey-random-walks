import matplotlib.pyplot as plt
from descartes import PolygonPatch
import geotiler
from heatmaps_common import *

if __name__ == '__main__':
    polys, nbrhoods = GetPolysNbrhoods('data/neighborhoods_Manhattan.geojson')
    nbrhood_keys = ReadKeys('processed_data/neighborhood_keys.pkl')
    seqs = GetSeqs('processed_data/manhattan-year-seqs.txt')
    bbox = GetBBox(polys, nbrhoods)
    alpha_map = AlphaMap(seqs, nbrhood_keys, logscale=False)

    # Display the image
    fig = plt.figure()
    ax = fig.gca()

    z = 12
    toner = geotiler.find_provider('stamen-toner-lite')
    mm = geotiler.Map(extent=bbox, zoom=z, provider=toner)
    img = geotiler.render_map(mm)
    img.save('test.png')
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for poly, nbrhood in zip(polys, nbrhoods):
        coords = poly['coordinates'][0]
        poly['coordinates'][0] = [mm.rev_geocode(p) for p in coords]
        patch = PolygonPatch(poly, alpha=-1.0 / np.log(alpha_map[nbrhood]))
        ax.add_patch(patch)

    plt.savefig('taxi-distribution.png', dpi=200, bbox_inches='tight')
    plt.show()
