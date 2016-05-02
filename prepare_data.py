import pandas as pd
import numpy as np

top_crimes = [
    'LARCENY/THEFT',\
    'OTHER OFFENSES',\
    'NON-CRIMINAL',\
    'ASSAULT',\
    'DRUG/NARCOTIC',\
    'VEHICLE THEFT',\
    'VANDALISM',\
    'WARRANTS',\
    'BURGLARY',\
    'SUSPICIOUS OCC',\
    'MISSING PERSON',\
    'ROBBERY',\
    'FRAUD',\
    'FORGERY/COUNTERFEITING',\
    'SECONDARY CODES']

def HourOfDay(df):
    ft = df.map(lambda x: x.hour)
    feat = pd.cut(ft, bins = [0,3,6,9,12,15,18,21,24], right=False)
    feat.name = 'HourOfDay'
    return feat

def Month(df):
    feat = df.map(lambda x: x.strftime("%B"))
    feat.name = 'Month'
    return feat

def ZipCode(df):
    feat = df.map(lambda x: 'Z'+str(x))
    feat.name = 'ZipCode'
    return feat



sf_crime = pd.read_csv('data/train.csv', quotechar='"')
sf_locs = pd.read_csv('data/sf_locations_zip.csv') 

sf_crime['Location'] = sf_crime['Y'].astype(str).str.cat(sf_crime['X'].astype(str), sep=',')
data = sf_crime.merge(sf_locs[['Location','ZipCode']], how='left', on='Location')

data['DateTime'] = pd.to_datetime(data['Dates'])
data['Output'] = data['Category']
data.loc[~data['Output'].isin(top_crimes), 'Output'] = 'MINOR CRIMES'
data.shape

dataset = pd.DataFrame(index=data.index)

dataset = dataset.join(data['Output'])
##dataset = dataset.join(data['DateTime'])
dataset = dataset.join(data['DayOfWeek'])
dataset = dataset.join(HourOfDay(data['DateTime']))
dataset = dataset.join(Month(data['DateTime']))
dataset = dataset.join(data['PdDistrict'])
dataset = dataset.join(ZipCode(data['ZipCode']))

rn = np.random.rand(len(dataset))
msk1 = rn <= 0.6
msk2 = (rn > 0.6) & (rn <= 0.8)
msk3 = rn > 0.8 

dataset[msk1].to_csv('data/train-A.csv', index=False)
dataset[msk2].to_csv('data/train-B.csv', index=False)
dataset[msk3].to_csv('data/train-C.csv', index=False)
