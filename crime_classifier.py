
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from pandas_confusion import ConfusionMatrix

get_ipython().magic('matplotlib inline')


# In[2]:

def HourOfDay(df):
    ft = df.map(lambda x: x.hour)
    feat = pd.cut(ft, bins = [0,3,6,9,12,15,18,21,24], right=False)
    feat.name = 'HourOfDay'
    return feat

def Month(df):
    feat = df.map(lambda x: x.strftime("%B"))
    feat.name = 'Month'
    return feat

def Week(df):
    feat = df.apply(lambda x: 'W'+str(x.isocalendar()[1]))
    feat.name = 'Week'
    return feat

def ZipCode(df): 
    feat = df.apply(lambda x: 'Z'+str(x))
    feat.name = 'ZipCode'
    return feat

def IsIntersection(df):
    feat = df.apply(lambda x: 0 if x.find("/") < 0 else 1)
    feat.name = 'IsIntersection'
    return feat

def Lng(df):
    minX, maxX = df.min(), df.max()
    feat = df.apply(lambda x: (x-minX)/(maxX-minX))
    feat.name = 'Lng'
    return feat

def Lat(df):
    minY, maxY = df.min(), df.max()
    feat = df.apply(lambda x: (x-minY)/(maxY-minY))
    feat.name = 'Lat'
    return feat

def build_dataset(data, n=14):
    data['Dates'] = pd.to_datetime(data['Dates'])
    if 'Category' in data.columns:
        topcats = data['Category'].value_counts().index.tolist()[n:]
        data['Target'] = data['Category']
        data.loc[data['Target'].isin(topcats), 'Target'] = 'OTHER CRIMES'

    features = pd.DataFrame(index=data.index)
    features = features.join(data['DayOfWeek'])
    features = features.join(HourOfDay(data['Dates']))
    features = features.join(Month(data['Dates']))
    features = features.join(data['PdDistrict'])
    features = features.join(ZipCode(data['ZipCode']))
    features = features.join(Lat(data['Y']))
    features = features.join(Lng(data['X']))
    features = features.join(IsIntersection(data['Address']))
    features = features.join(Week(data['Dates']))
        
    return features, data['Target'] if 'Target' in data.columns else None


# In[3]:

train_data = pd.read_csv('data/train-A.csv', quotechar='"')
test_data  = pd.read_csv('data/train-B.csv', quotechar='"')


# In[4]:

train_features,y_train  = build_dataset(train_data)
test_features,y_test    = build_dataset(test_data)


# In[5]:

y_train.value_counts()/sum(y_train.value_counts())


# In[6]:

train_features.head(5)


# In[7]:

train_features.shape


# In[8]:

d_train = train_features.T.to_dict().values()
d_test = test_features.T.to_dict().values()

vectorizer = DictVectorizer(sparse=True)

X_train = vectorizer.fit_transform(d_train)
X_test  = vectorizer.transform(d_test)

X_train.shape


# In[9]:

vectorizer.get_feature_names()


# In[ ]:

rparams = dict(C=[1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2])

lr = LogisticRegression(penalty='l2', C=0.01, max_iter=1000,                         multi_class='ovr', n_jobs=-1,                         solver='lbfgs', class_weight='balanced', verbose=0) ## , 

clf = GridSearchCV(lr, param_grid=rparams, scoring='log_loss', n_jobs=1, verbose=2)
clf.fit(X_train, y_train)


# In[ ]:

print(clf.best_params_)


# In[ ]:

cv_scores = np.zeros(6)
c = np.zeros(6)

rownum = 0
for params, mean_score, scores in clf.grid_scores_:
    c[rownum] = params['C']
    cv_scores[rownum] = -1.0*mean_score
    rownum = rownum + 1

plt.plot(c, cv_scores, '-o')
plt.xscale('log')


# In[ ]:

predicted = clf.predict(X_test)
expected = y_test
print(accuracy_score(expected, predicted))


# In[ ]:

predicted_probs = clf.predict_proba(X_test)
print(log_loss(y_test, predicted_probs))


# In[ ]:

cm = ConfusionMatrix(expected, predicted)
cm_stats = cm.to_dataframe().apply(lambda x: x/sum(x), axis=1)
cm_stats.to_csv('data/confusion_matrix_stats.csv')


# In[ ]:

mpl.rcParams['figure.figsize'] = (10.0, 5.0)
cm.plot(normalized=True)

