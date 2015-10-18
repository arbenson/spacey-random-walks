import json
from shapely.geometry import shape, Point
from collections import defaultdict
import csv
import time

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

def FindNeighborhood(lat, long):
    ''' Given a latitude and longitude, return the neighborhood. '''
    pt = Point(long, lat)
    for poly, nbrhood in nbrhood_polys:
        if poly.contains(pt):
            return nbrhood
    return 'OTHER'

def datetime(dt):
    ''' Return time struct from time string in data files. '''
    return time.strptime(dt, '%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    seqs = defaultdict(list)

    with open('data/trip_data_1.csv', 'rb') as csvfile:
        trips = csv.DictReader(csvfile)
        for trip in trips:
            lat = float(trip['pickup_latitude'])
            long = float(trip['pickup_longitude'])
            pickup_nbrhood = FindNeighborhood(lat, long)
            pickup_time = datetime(trip['pickup_datetime'])
            
            lat = float(trip['dropoff_latitude'])
            long = float(trip['dropoff_longitude'])
            dropoff_nbrhood = FindNeighborhood(lat, long)
            dropoff_time = datetime(trip['dropoff_datetime'])

            medallion = trip['medallion']

            seqs[medallion].append((pickup_time, pickup_nbrhood, dropoff_nbrhood))

        for medallion in seqs:
            seq = seqs[medallion]
            seq.sort()
            print [x[1] for x in seq]
    



