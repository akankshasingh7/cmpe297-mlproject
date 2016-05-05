import pandas as pd
import numpy as np

sf_crime = pd.read_csv('data/train.csv', quotechar='"')
print(sf_crime.shape)
sf_locs = pd.read_csv('data/sf_locations_zip.csv') 

sf_crime['Location'] = sf_crime['Y'].astype(str).str.cat(sf_crime['X'].astype(str), sep=',')
data = sf_crime.merge(sf_locs[['Location','ZipCode']], how='left', on='Location')
data.drop('Location', axis=1, inplace=True)

topcats = data['Category'].value_counts().index.tolist()[14:]
data['Target'] = data['Category']
data.loc[data['Target'].isin(topcats), 'Target'] = 'OTHER CRIMES'

rn = np.random.rand(len(data))
msk = rn <= 0.7

data[msk].to_csv('data/train-A.csv', index=False)
data[~msk].to_csv('data/train-B.csv', index=False)
