import csv
import sys

# Print all medallions from the raw trajectory data.
#
# USAGE:
#    python extract_medallions.py ../data/trip_data_1.csv > medallions.txt

if __name__ == '__main__':
    all_medallions = {}
    with open(sys.argv[1], 'rb') as csvfile:
        trips = csv.DictReader(csvfile)
        for i, trip in enumerate(trips):
            try:
                medallion = trip['medallion']
                all_medallions[medallion] = 1
            except:
                continue
                
            if i % 100000 == 0:
                sys.stderr.write(str(i) + '\n')

    for key in all_medallions:
        sys.stdout.write(key + '\n')
