#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:58:40 2017

@author: flavio
"""

#%% Libraries and data to be imported 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats as stats

os.chdir('/home/flavio/Dengue_fever')

X= pd.read_csv('data/dengue_features_train.csv')
y= pd.read_csv('data/dengue_labels_train.csv')
# y dataframe contains 3 columns. The last one is the column of interest containing
# the total dengue fever cases. Merge to create 1 dataset for analysis

Xy= pd.merge(X,y, how='inner', left_on=['city', 'year', 'weekofyear'], 
             right_on=['city','year','weekofyear'])

# converting week_date to date in a new column
Xy['week_start_date_dt']=pd.to_datetime(Xy['week_start_date'])

# create a month column
import datetime as dt
Xy['month']= Xy.week_start_date_dt.dt.month

#%% Some data manipulation
Xy.shape
Xy.describe()
Xy.dtypes
Xy.groupby(['city']).agg({'weekofyear':np.size})

# Check for missing data:
Xy.apply(lambda x: sum(x.isnull()),axis=0)


Xy2 = Xy.copy()

#%% Outlier Detection to trim data set
import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

out=outliers_iqr(Xy2.total_cases)

# trim the outliers
Xy2= Xy2[Xy2.total_cases <= 100]

Xy2.apply(lambda x: sum(x.isnull()), axis=0)

#%%
'''Not many missing values, but a good way to do missing value imputation
is to inpute the mean of city/month - this way we get an average per location
'''
 
col=Xy2.describe().columns #just num col titles
for c in col:
    Xy2[c].fillna(Xy2.groupby(['city','month'])[c].transform('median'), inplace=True)
    
    
#sanity check if replacement worked
Xy.groupby(['city','month']).agg({'station_avg_temp_c':np.median})
Xy2[['city', 'month', 'station_avg_temp_c']][Xy.station_avg_temp_c.isnull()]

from pandasql import sqldf
sqldf('select city, month, avg(station_avg_temp_c) from Xy group by city, month')

Xy2.apply(lambda x: sum(x.isnull()), axis=0)
# no more missing values

#%% Descriptive statistics

cor= Xy2.corr().reset_index()
#only correlations above abs(0.7)
cor[cor >= 0.7]
cor[cor <= -.7]

#only correlations with total cases
cor[['index', 'total_cases']]

sns.regplot(x='reanalysis_min_air_temp_k', y='total_cases',  data=Xy2, fit_reg=True)
plt.plot(Xy2.station_max_temp_c, Xy2.total_cases, 'bo')

# plotting months
plt.plot(Xy2.month, Xy2.total_cases, 'bo')
sns.countplot(x='month', data=Xy2)
# it seems that towards the end of the year (summer) cases are more frequent
# we have similar data points across all months.

# avg temperature seems around 28 in the entire dataset. 
Xy2.station_avg_temp_c.hist(bins=100)
Xy2.station_avg_temp_c.mean()

#Average per city:
Xy2.station_avg_temp_c[Xy2.city=='sj'].mean()
Xy2.station_avg_temp_c[Xy2.city=='sj'].hist(bins=100)

Xy2.station_avg_temp_c[Xy2.city=='iq'].mean()
Xy2.station_avg_temp_c[Xy2.city=='iq'].hist(bins=100)

# temperature are normally distributed and have similar means



#%% Hypothesis Tests
# ANOVA TEST
anova=smf.ols(formula='total_cases ~ C(month)',data=Xy2).fit()
anova.summary()

anova=smf.ols(formula='total_cases ~ C(city)',data=Xy2).fit()
anova.summary()


# at least 1 of the means is different according to the anova test
Xy2.groupby(['city']).agg({'total_cases':np.mean})

Xy2.boxplot('total_cases', by='city')
Xy2.boxplot('total_cases', by='month')

# Perform a post hoc test: Tukey Honest Significance Diff HSD
mc= multi.MultiComparison(Xy2['total_cases'], Xy2['month'])
res= mc.tukeyhsd()
print(res.summary())

# Pearson corr
Xy2.corr() # gives us only corr coefs. pearson r gives pvalues
stats.pearsonr(Xy2.reanalysis_min_air_temp_k, Xy2.total_cases)
stats.pearsonr(Xy2.year, Xy2.total_cases)

sns.regplot(x='year', y='total_cases', data=Xy2)
plt.plot(Xy2.reanalysis_air_temp_k, Xy2.total_cases, 'bo')


#%% Lineear regression model 
'''
some variables that are highly correlated with each other and may not be
a good idea to include in the model:
- ndvi_ne, ndvi_nw, ndvi_se, ndvi_sw
- month x week_of_year
- reanalysis_sat_precip_amt_mm x precipitation_amt_mm (perf corr)
- reanalysis_specific_humidity_g_per_kg x reanalysis_dew_point_temp_k


Interesting variables for the model:
'city', 'year', 'weekofyear', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm', 'total_cases'

Best approach will be to create 2 datasets, one for each city and run
the model on each. The Linear model can help to explain the role of the
variables and a better model can be tried later on.
'''
import statsmodels.api as sm
import statsmodels.formula.api as smf

reg1= smf.ols('''total_cases ~ C(month) + precipitation_amt_mm+ reanalysis_air_temp_k+
        reanalysis_dew_point_temp_k+
       reanalysis_max_air_temp_k+ reanalysis_min_air_temp_k+
       reanalysis_relative_humidity_percent+ reanalysis_tdtr_k+
       station_avg_temp_c+   station_min_temp_c + C(year) + C(city)''', data= Xy2).fit()
print(reg1.summary())

# QQ plot of normality of errors
sm.qqplot(reg1.resid, line='r')

# Plot of residuals
stdres= pd.DataFrame(reg1.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# additional regression diagnostic plots
fig=plt.figure(figsize=(12, 8))
sm.graphics.plot_regress_exog(reg1,  "precipitation_amt_mm", fig=fig)

# leverage plot
sm.graphics.influence_plot(reg1, size=8)

#%% Gradient Boost Decision Tree (a Random Forest Model)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale as scale
from sklearn.model_selection import train_test_split

#convert city to numeric
city_dic = {'iq':0, 'sj':1}
Xy2['city_cd']= Xy2['city'].map(city_dic)

X= Xy2[['city_cd', 'year', 'weekofyear',   'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']]

y= Xy2['total_cases']

Xs = X.copy()
cols = Xs.columns

for c in cols:
    Xs[c] = scale(Xs[c].astype('float64'))

X_train, X_test, y_train, y_test = train_test_split(Xs,y, random_state=0)

gbr= GradientBoostingRegressor(max_depth=6).fit(X_train, y_train)
gbr.score(X_train, y_train)
gbr.score(X_test, y_test)

# Plot of feature importances
def plot_feature_importances(model, train_data, feature_names):
    n_features = train_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


print(gbr.feature_importances_)
f_names = X.columns
plot_feature_importances(gbr, X_train, f_names)

# predict on the test data set
test= pd.read_csv('data/dengue_features_test.csv')

# treat missing values 
test.apply(lambda x: sum(x.isnull()), axis=0)

col=test.describe().columns #just num col titles
 
for c in col:
    test[c].fillna(Xy2.groupby(['city','month'])[c].transform('mean'), inplace=True)
    
city_dic = {'iq':0, 'sj':1}
test['city_cd']= test['city'].map(city_dic)   


test2= test[['city_cd', 'year', 'weekofyear',   'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']]

cols = test2.columns

for c in cols:
    test2[c] = scale(test2[c].astype('float64'))

pred = gbr.predict(test2)

sub1= test[['city','year','weekofyear']]
tmp= pd.DataFrame(pred)
sub2= pd.merge(sub1, tmp, left_index = True, right_index= True)
sub2=sub2.rename(columns={0: 'total_cases'})

sub2.to_csv('data/predictions.csv', index=False)


#%% New model 1 
''' baseline 
    train acc : 0.98638651944936795
    test acc: 0.63133040898740611
'''
cor= Xy2.corr()
cor[cor< -.7]

features= [ c for c in Xy2.columns if c not in [ 'city', 'week_start_date_dt',
            'week_start_date', 'total_cases']] 
X= Xy2[features]
y = Xy2.total_cases

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

gbr= GradientBoostingRegressor(max_depth=5, n_estimators=100).fit(X_train, y_train)
gbr.score(X_train, y_train)
gbr.score(X_test, y_test)

#%% grid search
from sklearn.model_selection import GridSearchCV
param_test = {'max_depth':range(5, 10), 'n_estimators': range(100, 500, 100)}
gsearch= GridSearchCV(estimator=GradientBoostingRegressor(), 
                      param_grid= param_test,  n_jobs=4, cv=5)

# even if it performs cross validation, we want to use a holdout to test
gsearch.fit(X_train,y_train)

gsearch.grid_scores_
gsearch.best_params_
gsearch.best_score_

final= GradientBoostingRegressor(max_depth=5, n_estimators=100).fit(X_test, y_test)
final.score(X_test, y_test)
