#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:59:01 2023

@author: chien
"""

######## Import Modules ########

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE # not possible to import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc,  recall_score, accuracy_score, precision_score, confusion_matrix,average_precision_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import shap
shap.initjs()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import folium
from folium.plugins import MarkerCluster

######## Pre-processing ########

df = pd.read_csv('/Users/chien/Library/CloudStorage/OneDrive-McGillUniversity/7c_INSY_662/Group Project/Crash_Reporting_-_Drivers_Data.csv')

# to check for na values
df.info()
df.isna().sum()

# dropping identifier and location columns
df.drop(['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID','Agency Name','Location'],axis=1,inplace=True)

# dropping columns with high number of missing values and columns that contain info that will be known after investigation
cols_to_drop = ['Off-Road Description', 'Municipality', 'Related Non-Motorist', 'Circumstance', 'Non-Motorist Substance Abuse','Driver Substance Abuse','Driver At Fault','Driver Distracted By']
df = df.drop(columns=cols_to_drop)
df.isna().sum()

# df report type
df['ACRS Report Type'].value_counts()

# convert date time column to datetime

df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'])
df['Hour'] = df['Crash Date/Time'].dt.hour
df['Day'] = df['Crash Date/Time'].dt.day
df['Month'] = df['Crash Date/Time'].dt.month
df['Year'] = df['Crash Date/Time'].dt.year

# cate the hour
# Define a dictionary to map months to seasons
seasons_mapping = {
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall',
    12: 'Winter', 1: 'Winter', 2: 'Winter'
}

# Create a new column 'Season' based on the 'Month' column
df['Season'] = df['Month'].map(seasons_mapping)

# Create a new column 'Time of Day' based on the 'Hour' column
conditions = [
    (df['Hour'] >= 6) & (df['Hour'] < 12),
    (df['Hour'] >= 12) & (df['Hour'] < 18),
    (df['Hour'] >= 18) & (df['Hour'] < 24),
    (df['Hour'] >= 0) & (df['Hour'] < 6)
]
choices = ['6am-12pm', '12pm-18pm', '18pm-24am', '0am-6am']
df['Time of Day'] = np.select(conditions, choices)

# Create a new column 'Month Segment' based on the 'Day' column
conditions = [
    (df['Day'] >= 1) & (df['Day'] <= 10),
    (df['Day'] >= 11) & (df['Day'] <= 20),
    (df['Day'] >= 21) & (df['Day'] <= 31)
]
choices = ['Beginning of Month', 'Mid Month', 'End of Month']
df['Month Segment'] = np.select(conditions, choices)

## drop na rows because of a large number of datasets
df2=df.dropna()
df2.shape

## count of severity
df2['Injury Severity'].value_counts()
df['Injury Severity'].value_counts()

## drop date time
df2.drop(['Crash Date/Time', 'Month', 'Day', 'Hour'], axis=1, inplace=True)


######## Exploratory Data Analysis (EDA) ########

#### colour and format setting
#light green, light blue, orange, light red, purple, grey
custom_palette = ["#8ACAAB", "#8CD2EE", "#ED9E58", "#F4B2B1","#6A63D9","#D3D3D3"]

# Set the custom color palette using Seaborn
sns.set_palette(custom_palette)

# Set Seaborn style for consistent aesthetics
sns.set(style="whitegrid")


## done on just pre processed data

## map analysis
# Create a base map
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)

# Add circle markers to the map
for index, row in df.iterrows():
   folium.CircleMarker(location=[row['Latitude'], row['Longitude']], 
                       radius=3, 
                       fill=True,
                       fill_opacity=0.6).add_to(m)

# Create a marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Add markers to the cluster
for index, row in df.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(marker_cluster)

# Display the map
m

## injury severity

plt.figure(figsize=(8, 6))

# Create the countplot
ax = sns.countplot(y='Injury Severity', data=df, order=df['Injury Severity'].value_counts().index)

# Display counts on the side of the bars
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.0f'), 
                (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points')

plt.title('Count of Injury Severity by Categories')
plt.savefig('Count_of_Injury_Severity_by_Categories.png', dpi=300, bbox_inches='tight')
plt.show()


## number of crashes by year

plt.figure(figsize=(20, 15))

# Count the crashes per year and plot as a line plot
df['Crash Date/Time'].dt.year.value_counts().sort_index().plot(kind='line',linewidth = 5,color='#F4B2B1')

plt.title('Number of Crashes by Year',fontsize=30)
plt.xlabel('Year', fontsize=30)
plt.ylabel('Number of Crashes', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Set y-axis lower limit to 0
plt.ylim(bottom=0)

plt.grid(True)
plt.savefig('Count_of_Crashes_by_Year.png', dpi=300, bbox_inches='tight')
plt.show()

## number of crashes by month/ season

plt.figure(figsize=(20, 15))

# Count the crashes per month and plot as a line plot
df['Crash Date/Time'].dt.month.value_counts().sort_index().plot(kind='line',linewidth = 5,color='#F4B2B1')

plt.title('Number of Crashes by Month',fontsize=30)
plt.xlabel('Month', fontsize=30)
plt.ylabel('Number of Crashes', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Set y-axis lower limit to 0
plt.ylim(bottom=0)

plt.grid(True)
plt.savefig('Count_of_Crashes_by_Month.png', dpi=300, bbox_inches='tight')
plt.show()


## number of crashes by hour
# accident hourly granularity and make it a line plot

# Count the crashes per month and plot as a line plot
df['Crash Date/Time'].dt.hour.value_counts().sort_index().plot(kind='line',linewidth = 5,color='#F4B2B1')

plt.title('Number of Crashes by Hour',fontsize=30)
plt.xlabel('Hour', fontsize=30)
plt.ylabel('Number of Crashes', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Set y-axis lower limit to 0
plt.ylim(bottom=0)

plt.grid(True)
plt.savefig('Count_of_Crashes_by_Hour.png', dpi=300, bbox_inches='tight')
plt.show()

## relationship with weather
plt.figure(figsize=(16, 10))
sns.countplot(y=df['Weather']);

## relationship with surface condition
plt.figure(figsize=(20, 15))

# Order bars by count
order = df['Surface Condition'].value_counts().index

# Plot
ax = sns.countplot(y=df['Surface Condition'], order=order, palette="viridis")

# Annotate bars with their counts
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), 
                va='center', ha='left', xytext=(5,0), textcoords='offset points', fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Number of Crashes',fontsize=30)
plt.ylabel('Surface Condition',fontsize=30)
plt.show()

## visualizing injury severity

plt.figure(figsize=(8, 6))

# Create the countplot
ax = sns.countplot(y='Injury Severity', data=df2, order=df2['Injury Severity'].value_counts().index)

# Display counts on the side of the bars
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.0f'), 
                (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points')

dummy_data = [plt.bar([0], [0], label=label) for i, label in enumerate(['No Injury', 'Injury'])]
plt.legend()
plt.title('Injury Class Distribution')
plt.savefig('Injury Class Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# drop observations with 'UNKNOWN' and 'Unknown' in the entire dataset
df2=df.dropna()
df2.replace(['UNKNOWN', 'Unknown'], np.nan, inplace=True)
df2.dropna(inplace=True)
df2.shape

## clustering data
model=KMeans(n_clusters=2)
model.fit(df2[['Longitude','Latitude']])
clusters=model.predict(df2[['Longitude','Latitude']])
plt.figure(figsize=(20, 15))
plt.scatter(df2['Longitude'],df2['Latitude'],c=clusters,s=10,cmap='viridis')
plt.title('Crash Locations Clustering',fontsize=30)

Fatal_Crashes=df[df['ACRS Report Type']=='Fatal Crash']
plt.scatter(Fatal_Crashes['Longitude'],Fatal_Crashes['Latitude'],color='red',s=10);

df2['location_cluster']=clusters
df2.drop(['Longitude','Latitude'],axis=1,inplace=True)
df2['location_cluster'].value_counts()
df2['location_cluster'].replace({0:'Location_cluster_1',1:'Location_cluster_2'},inplace=True)
df2['location_cluster'].value_counts()

# dropping the following columns because they have too many unique values; they most likely wont be good predictors, and some of them are identifiers: Veicle Make, Vehicle Model, Drivers License State, Roade Name,and Cross-Street Name
df2.drop(['Vehicle Make', 'ACRS Report Type', 'Vehicle Model', 'Drivers License State', 'Road Name', 'Cross-Street Name'], axis=1, inplace=True)

df2.replace({'Injury Severity': {'POSSIBLE INJURY': 1, 'SUSPECTED MINOR INJURY': 1, 'SUSPECTED SERIOUS INJURY': 1, 'FATAL INJURY': 1, 'NO APPARENT INJURY': 0}}, inplace=True)

## visualizing the composition of injury severity
df2['Injury Severity'].value_counts(normalize=True)
sns.countplot(y=df2['Injury Severity']);



## visualizing severity by location clusters:
plt.figure(figsize=(20, 15))
sns.countplot(x=df2['location_cluster'], hue=df2['Injury Severity'])
plt.title('Number of Crashes by Location Cluster')
plt.xlabel('Location Cluster')
plt.ylabel('Number of Crashes');

## visualizing severity by time of day
plt.figure(figsize=(20, 15))
sns.countplot(x=df2['Time of Day'], hue=df2['Injury Severity'])
plt.title('Number of Crashes by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Crashes');

## visualizing severity by time of month
plt.figure(figsize=(20, 15))
sns.countplot(x=df2['Time of Month'], hue=df2['Injury Severity'])
plt.title('Number of Crashes by Time of Month')
plt.xlabel('Time of Month')
plt.ylabel('Number of Crashes');

## visualizing severity by season
plt.figure(figsize=(20, 15))
sns.countplot(x=df2['Season'], hue=df2['Injury Severity'])
plt.title('Number of Crashes by Season')
plt.xlabel('Season')
plt.ylabel('Number of Crashes');

######## Model Building ########

## dummifying categorical columns
cols_to_dummy=df2.select_dtypes(include='object').columns
df2=pd.get_dummies(df2,columns=cols_to_dummy,drop_first=True,dtype='int')

## train_test_split
X = df2.drop("Injury Severity", axis=1)
y = df2["Injury Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

## Undersampling of negative class 
df_minority = df2[df2['Injury Severity']==1]
df_majority = df2[df2['Injury Severity']==0]

df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority),     # to match minority class
                                   random_state=42) 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
df_downsampled['Injury Severity'].value_counts()

#plot of downsampled

plt.figure(figsize=(8, 6))

# Create the countplot
ax = sns.countplot(y=df_downsampled['Injury Severity']);

# Display counts on the side of the bars
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.0f'), 
                (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points')

dummy_data = [plt.bar([0], [0], label=label) for i, label in enumerate(['No Injury', 'Injury'])]
plt.legend()
plt.title('Injury Class Distribution for Undersampled')
plt.savefig('Injury Class Distribution Undersampled.png', dpi=300, bbox_inches='tight')
plt.show()

X2 = df_downsampled.drop("Injury Severity", axis=1)
y2 = df_downsampled["Injury Severity"]
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X2, y2, test_size=0.20, random_state=42)

## Oversampling positive class
X3 = df2.drop('Injury Severity', axis=1)
y3 = df2['Injury Severity']
X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(X3, y3, test_size=0.20, random_state=42)


df_train = pd.concat([X_train_os, y_train_os], axis=1)


df_majority = df_train[df_train['Injury Severity'] == 0]  
df_minority = df_train[df_train['Injury Severity'] == 1]  

# Upsample the minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42)  


df_train_upsampled = pd.concat([df_majority, df_minority_upsampled])


df_train_upsampled = df_train_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)


X_train_os = df_train_upsampled.drop('Injury Severity', axis=1)
y_train_os = df_train_upsampled['Injury Severity']

df_train_upsampled['Injury Severity'].value_counts()
sns.countplot(y=df_train_upsampled['Injury Severity']);

## SMOTE resampling

X4 = df2.drop('Injury Severity', axis=1)  # features
y4 = df2['Injury Severity']                # target variable

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X4, y4, test_size=0.20, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

X_train_sm_resampled, y_train_sm_resampled = smote.fit_resample(X_train_sm, y_train_sm)

df_train_resampled = pd.DataFrame(X_train_sm_resampled, columns=X_train_sm.columns)
df_train_resampled['Injury Severity'] = y_train_sm_resampled

df_train_resampled = df_train_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_train_resampled['Injury Severity'].value_counts())
X_train_sm=df_train_resampled.drop('Injury Severity',axis=1)
y_train_sm=df_train_resampled['Injury Severity']

#### Model 1: Logistic Regression ####

## [normal] training and testing the model
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)
log_pred = logreg.predict(X_test)
print(classification_report(y_test, log_pred,target_names=['No Injury','Injury']))

## [normal] coefficients and intercept for the top features in a dataframe
coef = pd.DataFrame(logreg.coef_.T, index=X_train.columns, columns=['coef'])
coef['abs_coef'] = coef.coef.abs()
coef = coef.sort_values('abs_coef', ascending=False)
coef.drop('abs_coef', axis=1, inplace=True)
coef.head(30)

## [undersampled] training and testing the model
logreg_us = LogisticRegression(max_iter=10000)
logreg_us.fit(X_train_us, y_train_us)
log_pred_us = logreg_us.predict(X_test_us)
print(classification_report(y_test_us, log_pred_us,target_names=['No Injury','Injury']))

## [undersampled] coefficients and intercept for the top features in a dataframe
coef = pd.DataFrame(logreg_us.coef_.T, index=X_train_us.columns, columns=['coef'])
coef['abs_coef'] = coef.coef.abs()
coef = coef.sort_values('abs_coef', ascending=False)
coef.drop('abs_coef', axis=1, inplace=True)
coef.head(40)

## [oversampled] training and testing the model
logreg_os = LogisticRegression(max_iter=10000)
logreg_os.fit(X_train_os, y_train_os)
log_pred_os = logreg_os.predict(X_test_os)
print(classification_report(y_test_os, log_pred_os,target_names=['No Injury','Injury']))

## [oversampled] coefficients and intercept for the top features in a dataframe
coef = pd.DataFrame(logreg_os.coef_.T, index=X_train_os.columns, columns=['coef'])
coef['abs_coef'] = coef.coef.abs()
coef = coef.sort_values('abs_coef', ascending=False)
coef.drop('abs_coef', axis=1, inplace=True)
coef.head(30)

## [SMOTE] training and testing the model
logreg_sm = LogisticRegression(max_iter=10000)
logreg_sm.fit(X_train_sm, y_train_sm)
log_pred_sm = logreg_sm.predict(X_test_sm)
print(classification_report(y_test_sm, log_pred_sm,target_names=['No Injury','Injury']))

## [SMOTE] feature importance
coef = pd.DataFrame(logreg_sm.coef_.T, index=X_train_sm.columns, columns=['coef'])
coef['abs_coef'] = coef.coef.abs()
coef = coef.sort_values('abs_coef', ascending=False)
coef.drop('abs_coef', axis=1, inplace=True)
coef.head(30)

#### Model 2: Random Forest #####

## [normal] train and testing the model 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced',max_depth=5)
rf_model.fit(X_train, y_train)
rfc_predictions=rf_model.predict(X_test)
print(classification_report(y_test,rfc_predictions,target_names=['No Injury','Injury']))

## [normal] feature importance
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
print(feature_importances)

    # plot feature importance
(pd.Series(rf_model.feature_importances_, index=X.columns)
    .nlargest(30)  
    .plot(kind='barh', figsize=[16,10])
    .invert_yaxis()) 
plt.yticks(size=15)
plt.title('Most Important Features', size=18);


## [undersampled] train and testing the model 
rf_model_us = RandomForestClassifier(n_estimators=450, random_state=42, max_depth=11)
rf_model_us.fit(X_train_us, y_train_us)
rfc_predictions_us = rf_model_us.predict(X_test_us)
print(classification_report(y_test_us, rfc_predictions_us,target_names=['No Injury', 'Injury']))

## [undersampled] feature importance
feature_importances_us = pd.Series(rf_model_us.feature_importances_, index=X_train_us.columns).sort_values(ascending=False)
print(feature_importances_us)
feature_importances_us.nlargest(30).plot(kind='barh', figsize=[16,10]).invert_yaxis()
plt.yticks(size=15)
plt.title('Most Important Features for RF (undersampled)', size=18)
plt.show()


## [oversampled] train and testing the model 
rf_model_os = RandomForestClassifier(n_estimators=450, random_state=42, max_depth=11)
rf_model_os.fit(X_train_os, y_train_os)
rfc_predictions_os = rf_model_os.predict(X_test_os)
print(classification_report(y_test_os, rfc_predictions_os,target_names=['No Injury', 'Injury']))

## [oversampled] feature importance


## [SMOTE] train and test model
rf_model_sm = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=9)
rf_model_sm.fit(X_train_sm, y_train_sm)
rfc_predictions_sm = rf_model_sm.predict(X_test_sm)
print(classification_report(y_test_sm, rfc_predictions_sm,target_names=['No Injury', 'Injury']))

## [SMOTE] feature importance




#### Model 3: XGB ####

## [normal] training and testing the model

# assigning weight to balance the dataset
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, scale_pos_weight=scale_pos_weight,tree_method='hist')
xgb_model.fit(X_train, y_train)
xgb_predictions=xgb_model.predict(X_test)
print(classification_report(y_test,xgb_predictions,target_names=['No Injury','Injury']))    

## [normal] feature importance
feature_importances = pd.DataFrame(xgb_model.feature_importances_, index=X_train_us.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(50)

## [undersampled] training and testing the model
xgb_model_us = XGBClassifier(n_estimators=200, random_state=42, max_depth=5,tree_method='hist',max_bin=800,alpha=0.3,learning_rate=0.15)
xgb_model_us.fit(X_train_us, y_train_us)
xgb_predictions_us = xgb_model_us.predict(X_test_us)
print(classification_report(y_test_us, xgb_predictions_us, target_names=['No Injury', 'Injury']))

## [undersampled] feature importance
feature_importances_us = pd.DataFrame(xgb_model_us.feature_importances_, index=X_train_us.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances_us.head(50)

## removing features with zero importance
unused_features = list(feature_importances_us[feature_importances_us['importance'] == 0].index)
print(unused_features)
X_train_us.drop(unused_features, axis=1, inplace=True)
X_test_us.drop(unused_features, axis=1, inplace=True)


## [undersampled 2] training and testing the model
xgb_model_us_2 = XGBClassifier(n_estimators=200, random_state=42, max_depth=5, tree_method='hist', max_bin=800, alpha=0.3, learning_rate=0.15)
xgb_model_us_2.fit(X_train_us, y_train_us)
xgb_predictions_us_2 = xgb_model_us_2.predict(X_test_us)
print(classification_report(y_test_us, xgb_predictions_us_2, target_names=['No Injury', 'Injury']))

## [undersampled 2] feature importance
feature_importances_us_2 = pd.DataFrame(xgb_model_us_2.feature_importances_, index=X_train_us.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances_us_2.head(30)


## [oversampled] training and testing the model
xgb_model_os = XGBClassifier(n_estimators=200, random_state=42, max_depth=5,tree_method='hist',max_bin=800,alpha=0.3,learning_rate=0.15)
xgb_model_os.fit(X_train_os, y_train_os)
xgb_predictions_os = xgb_model_os.predict(X_test_os)
print(classification_report(y_test_os, xgb_predictions_os,target_names=['No Injury', 'Injury']))

## [oversampled] feature importance
feature_importances = pd.DataFrame(xgb_model_os.feature_importances_, index = X_train_os.columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)


## [SMOTE] train and test model
xgb_model_sm = XGBClassifier(n_estimators=500, random_state=42, max_depth=9,tree_method='hist',max_bin=800,alpha=0.3,learning_rate=0.15)
xgb_model_sm.fit(X_train_sm, y_train_sm)
xgb_predictions_sm = xgb_model_sm.predict(X_test_sm)
print(classification_report(y_test_sm, xgb_predictions_sm,target_names=['No Injury', 'Injury']))

## [SMOTE] feature importance
xgb_feature_importances = pd.DataFrame(xgb_model_sm.feature_importances_, index = X_train_sm.columns, columns=['importance']).sort_values('importance',ascending=False)
xgb_feature_importances.head(30)

unused_features = xgb_feature_importances[xgb_feature_importances['importance'] == 0]
unused_features

    # plot importance
(pd.Series(xgb_model_sm.feature_importances_, index=X.columns)
    .nlargest(20)  
    .plot(kind='barh', figsize=[20,15])
    .invert_yaxis()) 
plt.yticks(size=15)
plt.title('Most Important Features by XGBoost (SMOTE)', size=18);



######## Cross Comparison of the 3 Models ########

# Calculate metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred),
}

def calculate_metrics(y_true, y_pred):
    return {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

xgb_metrics = calculate_metrics(y_test, xgb_predictions)
rfc_metrics = calculate_metrics(y_test, rfc_predictions)
log_metrics = calculate_metrics(y_test, log_pred)
xgb_metrics_us = calculate_metrics(y_test_us, xgb_predictions_us)
rfc_metrics_us = calculate_metrics(y_test_us, rfc_predictions_us)
log_metrics_us = calculate_metrics(y_test_us, log_pred_us)
xgb_metrics_os = calculate_metrics(y_test_os, xgb_predictions_os)
rfc_metrics_os = calculate_metrics(y_test_os, rfc_predictions_os)
log_metrics_os = calculate_metrics(y_test_os, log_pred_os)
xgb_metrics_sm = calculate_metrics(y_test_sm, xgb_predictions_sm)
rfc_metrics_sm = calculate_metrics(y_test_sm, rfc_predictions_sm)
log_metrics_sm = calculate_metrics(y_test_sm, log_pred_sm)

comparison_df = pd.DataFrame({
    'XGBoost': xgb_metrics,
    'XGBoost (undersampled)': xgb_metrics_us,
    'XGBoost (oversampled)': xgb_metrics_os,
    'XGBoost (SMOTE)': xgb_metrics_sm,
    'Random Forest': rfc_metrics,
    'Random Forest (undersampled)': rfc_metrics_us,
    'Random Forest (oversampled)': rfc_metrics_os,
    'Random Forest (SMOTE)': rfc_metrics_sm,
    'Logistic Regression': log_metrics,
    'Logistic Regression (undersampled)': log_metrics_us,
    'Logistic Regression (oversampled)': log_metrics_os,
    'Logistic Regression (SMOTE)': log_metrics_sm
})

print(comparison_df)

######## Cross Validation and Hyperparameters Tuning ########

## Grid Search CV 
kf=KFold(n_splits=5,shuffle=True,random_state=42)
xgb_parameters = {
    'n_estimators': stats.randint(100, 1000),
    'max_depth': stats.randint(1, 11),
    'learning_rate': stats.uniform(0.01, 0.99),
    'max_bin' : stats.randint(100,1000),
    'alpha': stats.uniform(0.0, 100.0),
    'min_child_weight': stats.uniform(0, 10),
    'colsample_bytree': stats.uniform(0.5, 0.5),
    'gamma': stats.uniform(0, 10),
    'subsample': stats.uniform(0.5, 0.5)
}

xgbr = XGBClassifier(random_state=42,tree_method='hist')
xgb_grid_search = RandomizedSearchCV(xgbr, xgb_parameters, cv=kf, scoring='f1', n_jobs=-1, verbose=1)
xgb_grid_search.fit(X_train_us, y_train_us)

# Best RF Model
best_xgb = xgb_grid_search.best_estimator_

# RF predictions
xgb_preds = best_xgb.predict(X_test_us)
best_params = xgb_grid_search.best_params_

print(classification_report(y_test_us, xgb_preds))
print(f'Best parameters: {xgb_grid_search.best_params_}')
print(f'Best score: {xgb_grid_search.best_score_}')

## Apply parameters to model
# applying threshold moving to the best model, which is XGBoost Undersampled
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits,random_state=42,shuffle=True)

# Placeholder for the best threshold
best_thresholds = []
best_f1_scores = []
X2.drop(unused_features.index, axis=1, inplace=True)
# Selecting top 10 features

# Loop over each fold
for train_index, test_index in skf.split(X2, y2):
    X_train_th, X_test_th = X2.iloc[train_index], X2.iloc[test_index]
    y_train_th, y_test_th = y2.iloc[train_index], y2.iloc[test_index]

    # Train the model
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train_th, y_train_th)

    # Predict probabilities
    xgb_prob = xgb_model.predict_proba(X_test_th)[:, 1]

    # Find the best threshold for this fold
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0, 1, 101)
    for th in thresholds:
        preds = (xgb_prob >= th).astype(int)
        f1 = f1_score(y_test_th, preds)  # Corrected to use y_test_th
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    best_thresholds.append(best_threshold)
    best_f1_scores.append(best_f1)

# Calculate the average best threshold
final_best_threshold = np.mean(best_thresholds)
final_best_f1 = np.mean(best_f1_scores)

print(f'Average Best Threshold: {final_best_threshold}')
print(f'Average Best F1 Score: {final_best_f1}')

# Retrain the model on the entire dataset
xgb_model_final = XGBClassifier(**best_params)
xgb_model_final.fit(X2, y2)

# Final prediction 
xgb_prob_final = xgb_model_final.predict_proba(X2)[:, 1]
xgb_final_predictions = (xgb_prob_final >= final_best_threshold).astype(int)

# Final evaluation report
print('------------------------------------------------------------')
print(f'Final Model Performance with Threshold of {final_best_threshold}')
print(classification_report(y2, xgb_final_predictions, target_names=['No Injury', 'Injury']))

######## Comparison with other models ########
metrics = { 
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred),
}

def calculate_metrics(y_true, y_pred):
    return {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

xgb_metrics = calculate_metrics(y_test, xgb_predictions)
rfc_metrics = calculate_metrics(y_test, rfc_predictions)    
log_metrics = calculate_metrics(y_test, log_pred)
xgb_metrics_us = calculate_metrics(y_test_us, xgb_predictions_us)
rfc_metrics_us = calculate_metrics(y_test_us, rfc_predictions_us)
log_metrics_us = calculate_metrics(y_test_us, log_pred_us)
xgb_metrics_os = calculate_metrics(y_test_os, xgb_predictions_os)
rfc_metrics_os = calculate_metrics(y_test_os, rfc_predictions_os)
log_metrics_os = calculate_metrics(y_test_os, log_pred_os)
xgb_metrics_sm = calculate_metrics(y_test_sm, xgb_predictions_sm)
rfc_metrics_sm = calculate_metrics(y_test_sm, rfc_predictions_sm)
log_metrics_sm = calculate_metrics(y_test_sm, log_pred_sm)
xgb_metrics_final = calculate_metrics(y2, xgb_final_predictions)

comparison_df = pd.DataFrame({
    'XGBoost': xgb_metrics,
    'XGBoost (undersampled)': xgb_metrics_us,
    'XGBoost (oversampled)': xgb_metrics_os,
    'XGBoost (SMOTE)': xgb_metrics_sm,
    'XGBoost (0.38 Threshold )': xgb_metrics_final,
    'Random Forest': rfc_metrics,
    'Random Forest (undersampled)': rfc_metrics_us,
    'Random Forest (oversampled)': rfc_metrics_os,
    'Random Forest (SMOTE)': rfc_metrics_sm,
    'Logistic Regression': log_metrics,
    'Logistic Regression (undersampled)': log_metrics_us,
    'Logistic Regression (oversampled)': log_metrics_os,
    'Logistic Regression (SMOTE)': log_metrics_sm
})

print(comparison_df)

######## Optimization for Parsimony and Performance ########

# try to reduce number of features to increase parsimony
top_features = list(feature_importances.head(20).index)
X5 = X2[top_features]

# applying threshold moving to the best model, which is XGBoost Undersampled
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

# Placeholder for the best threshold
best_thresholds = []
best_f1_scores = []

# Loop over each fold using the dataset with top features
for train_index, test_index in skf.split(X5, y2):
    X_train_th, X_test_th = X5.iloc[train_index], X5.iloc[test_index]
    y_train_th, y_test_th = y2.iloc[train_index], y2.iloc[test_index]

    # Train the model with best parameters on the reduced feature set
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train_th, y_train_th)

    # Predict probabilities on the test set
    xgb_prob = xgb_model.predict_proba(X_test_th)[:, 1]

    # Find the best threshold for this fold
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0, 1, 101)
    for th in thresholds:
        preds = (xgb_prob >= th).astype(int)
        f1 = f1_score(y_test_th, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    best_thresholds.append(best_threshold)
    best_f1_scores.append(best_f1)

# Calculate the average best threshold
final_best_threshold = np.mean(best_thresholds)
final_best_f1 = np.mean(best_f1_scores)

print(f'Average Best Threshold: {final_best_threshold}')
print(f'Average Best F1 Score: {final_best_f1}')

# Retrain the model on the entire dataset with reduced features
xgb_model_final = XGBClassifier(**best_params)
xgb_model_final.fit(X5, y2)

# Final prediction using the model trained on reduced features
xgb_prob_final = xgb_model_final.predict_proba(X5)[:, 1]
xgb_final_predictions = (xgb_prob_final >= final_best_threshold).astype(int)

# Final evaluation report
print('------------------------------------------------------------')
print(f'Final Model Performance with Threshold of {final_best_threshold}')
print(classification_report(y2, xgb_final_predictions, target_names=['No Injury', 'Injury']))

######## Interpreting Model with LIME and SHAP ########
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X5.values,
    feature_names=X5.columns,
    class_names=['Non-Emergent', 'Emergent'],
    mode='classification',
    random_state=42
)
# Choose an instance 
instance_index = 102
instance = X5.iloc[instance_index].values

explanation = explainer.explain_instance(instance, xgb_model_final.predict_proba, num_features=20)

explanation.show_in_notebook(show_predicted_value=True)

explainer = shap.Explainer(xgb_model_final, X5,output_names=['Non-Emergent', 'Emergent'],seed=42)
shap_values = explainer.shap_values(X5)
# Summary plot
shap.summary_plot(shap_values, X5)


