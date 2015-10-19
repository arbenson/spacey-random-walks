from collections import defaultdict
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

    for feature in js['features']:
        poly = shape(feature['geometry'])
        nbrhood = feature['properties']['neighborhood']
        nbrhood_polys.append((poly, nbrhood))
    return nbrhood_polys

def FindNeighborhood(lat, long, nbrhood_polys):
    ''' Given a latitude and longitude, return the neighborhood. '''
    pt = Point(long, lat)
    for poly, nbrhood in nbrhood_polys:
        if poly.contains(pt):
            return nbrhood
    return 'OTHER'

def format_datetime(dt):
    ''' Return time struct from time string in data files. '''
    dt = datetime.fromtimestamp(mktime(strptime(dt, '%Y-%m-%d %H:%M:%S')))
    return (dt - datetime(2000,1,1)).total_seconds()

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

'''
tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender
'''

if __name__ == '__main__':
    seqs = defaultdict(list)

    for data_file in sys.argv[1:]:
        sys.stderr.write('Reading from %s...\n' % data_file)
        with open(data_file, 'rb') as csvfile:
            trips = csv.DictReader(csvfile)
            for i, trip in enumerate(trips):
                start_station = int(trip['start station id'])
                end_station = int(trip['end station id'])
                bikeid = trip['bikeid']
                start_time = format_datetime(trip['starttime'])
                seqs[bikeid].append((start_time, start_station, end_station))
                
                if i % 100000 == 0:
                    sys.stderr.write(str(i) + '\n')

    for _, seq in seqs.iteritems():
        sys.stdout.write(','.join([str(x) for x in ProcessSeq(seq)]) + '\n')
