import json
from shapely.geometry import shape, Point
import sys

keep_borough = sys.argv[1]

with open('neighborhoods.geojson', 'r') as f:
    nbrhoods_js = json.load(f)

new_features = []
for feature in nbrhoods_js['features']:
    p = shape(feature['geometry'])
    nbrhood = feature['properties']['neighborhood']
    borough = feature['properties']['borough']
    if borough == keep_borough:
        new_features.append(feature)
        print nbrhood

nbrhoods_js['features'] = new_features

with open('neighborhoods_%s.geojson' % keep_borough, 'w') as f:
    json.dump(nbrhoods_js, f)

