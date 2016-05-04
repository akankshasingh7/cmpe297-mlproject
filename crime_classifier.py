
# coding: utf-8

# In[51]:

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from pandas_confusion import ConfusionMatrix

## DayOfWeek,HourOfDay,Month,PdDistrict,ZipCode
feature_names = [ 'DayOfWeek',
                'HourOfDay',
                'Month',
                'PdDistrict',
                'ZipCode',
		'Address']

##get_ipython().magic('matplotlib inline')

# In[2]:

train_data = pd.read_csv('data/train-A.csv', quotechar='"')
validate_data  = pd.read_csv('data/train-B.csv', quotechar='"')
test_data  = pd.read_csv('data/train-C.csv', quotechar='"')

train_features,y_train  = train_data[feature_names], train_data['Output'] 
validate_features,y_validate  = validate_data[feature_names], validate_data['Output'] 
test_features,y_test  = test_data[feature_names], test_data['Output'] 

#class_counts = y_train.value_counts()
#class_prop = class_counts/sum(class_counts)
#class_wts = (1.0/class_prop) /sum(1.0/class_prop)
# In[3]:

d_train = train_features.T.to_dict().values()
d_validate = validate_features.T.to_dict().values()
d_test = test_features.T.to_dict().values()

vectorizer = DictVectorizer(sparse=True)

X_train = vectorizer.fit_transform(d_train)
## vectorizer.get_feature_names()
X_validate  = vectorizer.transform(d_validate)
X_test  = vectorizer.transform(d_test)

# In[66]:
#mdl = RandomForestClassifier(n_estimators=50, max_features=20, max_depth=10, min_samples_leaf=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
mdl = LogisticRegression(penalty='l2', C=0.1, max_iter=100, n_jobs=-1, solver='liblinear', verbose=0).fit(X_train, y_train)

# In[69]:

print(mdl)

# In[70]:

predicted = mdl.predict(X_train)
expected = y_train
print(accuracy_score(expected, predicted))

predicted = mdl.predict(X_test)
expected = y_test
print(accuracy_score(expected, predicted))

predicted_probs = mdl.predict_proba(X_test)
print(log_loss(y_test, predicted_probs))

# In[71]:
feature_imp = pd.Series(mdl.feature_importances_, index=vectorizer.get_feature_names(), name='Importance')
feature_imp.sort(ascending=False)
print(feature_imp)

cm = ConfusionMatrix(expected, predicted)
##print("Confusion matrix:\n%s" % cm)
##mpl.rcParams['figure.figsize'] = (10.0, 5.0)
cm.plot(normalized=True)
