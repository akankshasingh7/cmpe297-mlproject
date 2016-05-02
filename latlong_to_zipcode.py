import pandas as pd
import numpy as np
import csv
import time

from geopy.distance import great_circle
from pyzipcode import ZipCodeDatabase

zcdb = ZipCodeDatabase()
sf_zipcodes = dict([(z.zip, (z.latitude, z.longitude)) for z in zcdb.find_zip(city="San Francisco")])

def find_nearest_zip(loc):
    nearest_zip = '00000'
    min_dist = float('inf')
    for zipcode, latlon in sf_zipcodes.iteritems():
        distance = great_circle(loc, latlon).miles
        if  distance < min_dist:
            min_dist = distance
            nearest_zip = zipcode
    return nearest_zip

loc_dict = {}

sf_crime = pd.read_csv('data/train.csv') 
sf_crime['Location'] = sf_crime['Y'].astype(str).str.cat(sf_crime['X'].astype(str), sep=',')
np_locs = sf_crime['Location'].tolist()

for latlng in np_locs:
    loc_dict[latlng] = 1   

print 'Num locations=', len(loc_dict)

outfile = open('data/sf_locations_zip.csv', 'w')
writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Location', 'ZipCode', 'Center'])

num = 0
for latlon in loc_dict.keys():
    ll = latlon.split(',')
    zipcode = find_nearest_zip( (float(ll[0]), float(ll[1])) )
    center = sf_zipcodes.get(zipcode)
    writer.writerow([latlon, zipcode, center])
    #print ','.join(ll), zipcode
    num = num +1
    if num % 5000 == 0: print num, 'records processed'
