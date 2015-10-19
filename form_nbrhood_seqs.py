from collections import defaultdict
import cPickle as pickle
import csv
from datetime import datetime
import json
from shapely.geometry import shape, Point
import sys
from time import mktime, strptime

'''
Read the trip data and convert latlongs to a sequence of neighborhoods

http://stackoverflow.com/questions/20776205/point-in-polygon-with-geojson-in-python
'''

def GetNeighborhoods():
    ''' Return list of (shape polygon, neighborhood) tuples '''
    # load GeoJSON file containing NYC neighborhoods
    with open('data/neighborhoods.geojson', 'r') as f:
        js = json.load(f)

    # Note: there is more than one polygon per neighborhood
    nbrhood_polys = []
    nbrhood_keys = {0: 'OTHER'}

    for feature in js['features']:
        poly = shape(feature['geometry'])
        nbrhood = feature['properties']['neighborhood']
        if nbrhood not in nbrhood_keys:
            nbrhood_keys[nbrhood] = len(nbrhood_keys)
        nbrhood_polys.append((poly, nbrhood_keys[nbrhood]))
    return nbrhood_polys, nbrhood_keys

def FindNeighborhood(lat, long, nbrhood_polys):
    ''' Given a latitude and longitude, return the neighborhood. '''
    pt = Point(long, lat)
    for poly, nbrhood in nbrhood_polys:
        if poly.contains(pt):
            return nbrhood
    return 0

def format_datetime(dt):
    ''' Return time struct from time string in data files. '''
    dt = datetime.fromtimestamp(mktime(strptime(dt, '%Y-%m-%d %H:%M:%S')))
    return (dt - datetime(2012,1,1)).total_seconds()

def ProcessSeq(seq):
    ''' Process list of (pickup_time, pickup_nbrhood, dropoff_nbrhood) '''
    seq.sort()
    proc_seq = [seq[0][1], seq[0][2]]
    for i in xrange(1, len(seq)):
        curr_pickup = seq[i][1]
        curr_dropoff = seq[i][2]
        # Put the pickup in the sequence if it changed
        # from the last point.
        if curr_pickup != seq[i - 1][2]:
            proc_seq.append(curr_pickup)
        proc_seq.append(curr_dropoff)
    return proc_seq

if __name__ == '__main__':
    nbrhood_polys, nbrhood_keys = GetNeighborhoods()
    with open('neighborhood_keys.pkl', 'w') as f:
        pickle.dump(nbrhood_keys, f)
    seqs = defaultdict(list)

    for data_file in sys.argv[1:]:
        sys.stderr.write('Reading from %s...\n' % data_file)
        with open(data_file, 'rb') as csvfile:
            trips = csv.DictReader(csvfile)
            for i, trip in enumerate(trips):
                lat = float(trip['pickup_latitude'])
                long = float(trip['pickup_longitude'])
                pickup_nbrhood = FindNeighborhood(lat, long, nbrhood_polys)
                pickup_time = format_datetime(trip['pickup_datetime'])
                
                lat = float(trip['dropoff_latitude'])
                long = float(trip['dropoff_longitude'])
                dropoff_nbrhood = FindNeighborhood(lat, long, nbrhood_polys)
                
                medallion = trip['medallion']

                #sys.stdout.write('%s\t%s\t%s\t%s\n' % (
                #        medallion, pickup_nbrhood, dropoff_nbrhood, pickup_time))
                seqs[medallion].append((pickup_time, pickup_nbrhood, dropoff_nbrhood))
                
                if i % 100000 == 0:
                    sys.stderr.write(str(i) + '\n')

    for _, seq in seqs.iteritems():
        sys.stdout.write(','.join([str(x) for x in ProcessSeq(seq)]) + '\n')
