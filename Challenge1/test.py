# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:18:59 2022

@author: Lenovo
"""

from sklearn import linear_model
from statsmodels.nonparametric import kernel_regression
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import lars_path
from sklearn.linear_model import LassoLarsCV,LassoCV
from pygam import LinearGAM, s
from sklearn.cluster import KMeans

# Chargement des donnes
train_data = pd.read_csv("train_data.csv") 
test_data = pd.read_csv("test_data.csv")

def box_plot_outliers(data):
  q1,q3 = data.quantile(.25),data.quantile(.75)
  IQR = q3-q1
  low,up = q1 - 1.5*IQR, q3 + 1.5*IQR
  outlier = data.mask((data<low)|(data>up))
  return outlier
train_data['prix'] = pd.DataFrame(box_plot_outliers(train_data['prix']))
train_data.dropna(inplace = True)

zipcode = list(train_data['zipcode'].unique())

plot_prix = list()
for zip in zipcode:
  plot_prix.append(list(train_data.loc[train_data['zipcode'] == zip,['prix']].values.reshape(len(train_data.loc[train_data['zipcode'] == zip,['prix']]))))

# coor = train_data.loc[:,['lat','long','prix']]
# coor = coor.reset_index(drop = True)
# prix_sum = coor['prix'].mean()
# coor['prix'] = coor['prix'].map(lambda x:x/prix_sum * 80)
# coor_np = np.array(coor)

# n_clusters = 5          # Quantity of clusters
# estimator = KMeans(n_clusters, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)  # Construct K-means cluster
# estimator.fit(coor)

# print(estimator.cluster_centers_)# center

# quantity = pd.Series(estimator.labels_).value_counts()
# distortions = []

# res0Series = pd.Series(estimator.labels_) # individuals in each center
# t = train_data.copy(deep = True)
# t = t.reset_index(drop =True)
# t['label'] = res0Series

# def cal_dis(lan,lat,lan_c,lat_c):
#   return math.sqrt((lan-lan_c)**2+(lat-lat_c)**2)
# def decide_label(co,center):
#   dis = list()
#   for cen in center:
#     dis.append(cal_dis(co['long'],co['lat'],cen[1],cen[0]))
#   return dis.index(min(dis))


# ts = test_data.copy(deep = True)
# ts['label'] = ts.apply(lambda x:decide_label(x,estimator.cluster_centers_),axis = 1)

# train_data = t
# test_data = ts

# Extraire les variables
Xtrain = train_data.drop(['id','prix'],axis=1)
Xtest = test_data.drop(['id'],axis=1)
all_features = pd.concat((Xtrain,Xtest))
Ytrain = train_data['prix']

# Traitement des variables
# Convertir le temps en float
all_features['date'] = all_features['date'].map(lambda x:(str(x)).split('T')[0])
all_features['annee_vente'] = all_features['date'].map(lambda x:int(x.split('-')[0]))
all_features['date'] = all_features['date'].map(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d')))

# Annee_utilise
def set_annee_de_renovation(row):

  if(row['annee_renovation'] > row['annee_construction']):
    return row['annee_renovation']
  else:
    return row['annee_construction']

def set_annee_utilise(row):
  
  return abs(row['annee_vente'] - row['annee_utilise'])

all_features['annee_utilise'] = all_features.apply(lambda x:set_annee_de_renovation(x),axis = 1)
all_features['annee_utilise'] = all_features.apply(lambda x:set_annee_utilise(x), axis = 1)
all_features.drop(['annee_renovation','annee_construction','annee_vente'],axis = 1,inplace = True)

df_zipcode = (train_data.groupby(['zipcode'])['prix'].mean()).to_frame()
kmeans = KMeans(n_clusters = 6, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit(df_zipcode)
dic_zipcode = {}
for i in range(y_kmeans.labels_.shape[0]):
    dic_zipcode[df_zipcode.index[i]] = str(y_kmeans.labels_[i]+1)
all_features['zipcode'] = all_features['zipcode'].apply(
    lambda x:(dic_zipcode[x]))

# df_zipcode = (train_data.groupby(['zipcode'])['prix'].mean()).to_frame()
# group = []
# for i in range(21):
#     group.append(i*0.05)    
# df_quantile = df_zipcode.quantile(group, numeric_only=True)
# dic_zipcode = {}
# for i in range(df_zipcode.shape[0]):
#     for j in range(20):
#         if df_zipcode.iloc[i]['prix'] >= df_quantile.iloc[j]['prix'] and df_zipcode.iloc[i]['prix'] < df_quantile.iloc[j+1]['prix']:
#             dic_zipcode[df_zipcode.index[i]] = str(j+1)
#         elif df_zipcode.iloc[i]['prix'] == df_quantile.iloc[20]['prix']:
#             dic_zipcode[df_zipcode.index[i]] = str(20)

# all_features['zipcode'] = all_features['zipcode'].apply(
#     lambda x:(dic_zipcode[x]))

def to_str(data):
  return str(data)

# all_features = all_features.drop(['vue_mer','etat_note'],axis = 1)
# all_features['nb_chambres'] = pd.DataFrame(all_features['nb_chambres'].map(lambda x:to_str(x)))
# all_features['nb_sdb'] = pd.DataFrame(all_features['nb_sdb'].map(lambda x:to_str(x)))
# all_features['nb_etages'] = pd.DataFrame(all_features['nb_etages'].map(lambda x:to_str(x)))
# all_features['vue_note'] = pd.DataFrame(all_features['vue_note'].map(lambda x:to_str(x)))
# all_features['etat_note'] = pd.DataFrame(all_features['etat_note'].map(lambda x:to_str(x)))
# all_features['design_note'] = pd.DataFrame(all_features['design_note'].map(lambda x:to_str(x)))
# all_features['label'] = pd.DataFrame(all_features['label'].map(lambda x:to_str(x)))


# One-Hot Encoding pour "zipcode"
all_features = pd.get_dummies(all_features, dummy_na = True)

# Separer les variables de 'train' et 'test'
n_train = train_data.shape[0]
Xtrain = all_features.iloc[:n_train]
Xtest = all_features.iloc[n_train:]

# Obtenir index
numeric_features = Xtrain.dtypes[Xtrain.dtypes != 'object'].index

# # Normaliser les donnes par mean et std de train
# mean_Xtrain = Xtrain.mean()
# std_Xtrain = Xtrain.std()
# Xtrain[numeric_features] = Xtrain[numeric_features].apply(
#     lambda x:(x - x.mean())/(x.std()+0.0001)) 

# for i in range(numeric_features.shape[0]):
#     Xtest[numeric_features[i]] = Xtest[numeric_features[i]].apply(
#         lambda x:(x - mean_Xtrain[i])/(std_Xtrain[i]+0.0001))

# Definition of the root mean squared error
def rmse(y,yprime):
    return np.sqrt(mean_squared_error(y,yprime))
# Definition of the mean absolute percentage error
def mape(y,yprime):
    return 100 * np.sum(np.abs(y-yprime)/y)/len(y)
mape_error_synt = {}
rmse_error_synt = {}

# K-blocs CV
# Diviser les donnes
def get_k_blocs_data(k, i, X, Y):
    assert k > 1
    blocs_size = X.shape[0] // k
    X_train, Y_train = None, None
    for j in range(k):
        idx = slice(j * blocs_size, (j + 1) * blocs_size)
        X_part, Y_part = X[idx], Y[idx]
        if j == i:
            X_valid, Y_valid = X_part, Y_part
        elif X_train is None:
            X_train, Y_train = X_part, Y_part
        else:
            X_train = pd.concat([X_train, X_part])
            Y_train = pd.concat([Y_train, Y_part])
    return X_train, Y_train, X_valid, Y_valid 

# Régression linéaire multiple
def reglin(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train) 
        reglin = linear_model.LinearRegression() 
        reglin.fit(data[0],data[1]) 
        Y_prediction = reglin.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result1 = reglin(5, Xtrain, Ytrain)
rmse_error_synt['Multiple linear regression'] = result1[0]
mape_error_synt['Multiple linear regression']  = result1[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result1[0],result1[1]))

# Régression ridge
def ridge(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        cross_validation_ridge = RidgeCV(alphas = np.arange(0,4000,1)+0.1)
        cross_validation_ridge.fit(data[0],data[1])
        Y_prediction = cross_validation_ridge.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k
    
result2 = ridge(5, Xtrain, Ytrain)
rmse_error_synt['Ridge regression'] = result2[0]
mape_error_synt['Ridge regression']  = result2[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result2[0],result2[1]))

# Régression LASSO-LARS
def lassolars(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        Xtrain_lars_path = np.array(data[0],dtype='float64')
        Ytrain_lars_path = np.reshape(np.array(data[1],dtype='float64'),(data[1].shape[0],))
        path  = lars_path(Xtrain_lars_path,Ytrain_lars_path,method='lasso')
        xx = np.sum(np.abs(path[2].T), axis=1)
        xx /= xx[-1]
        plt.figure(figsize=(20,20))
        plt.plot(xx, path[2].T)
        ymin, ymax = plt.ylim()
        plt.vlines(xx, ymin, ymax, linestyle='dashed')  
        lasso_lars_cv = LassoLarsCV(cv = 4) 
        lasso_lars_cv.fit(Xtrain_lars_path,Ytrain_lars_path)
        Y_prediction = lasso_lars_cv.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result3 = lassolars(5, Xtrain, Ytrain)
rmse_error_synt['LASSO-LARS regression'] = result3[0]
mape_error_synt['LASSO-LARS regression']  = result3[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result3[0],result3[1]))

# Méthode de Nadaraya-Watson
def NW(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        kreg = kernel_regression.KernelReg(data[1],data[0]['m2_interieur'],"c")
        Y_prediction = kreg.fit(data[2]['m2_interieur'])[0]
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result4 = NW(5, Xtrain, Ytrain)
rmse_error_synt['N-W kernel regression'] = result4[0]
mape_error_synt['N-W kernel regression']  = result4[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result4[0],result4[1]))

# Generalized Additive Models
def GAM(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        spline = s(0)
        for j in range(1,39):
            spline += s(j)
        gam = LinearGAM(spline).fit(Xtrain,Ytrain)
        Y_prediction = gam.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result5 = GAM(5, Xtrain, Ytrain)
rmse_error_synt['GAM kernel regression'] = result5[0]
mape_error_synt['GAM kernel regression']  = result5[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result5[0],result5[1]))

plt.figure(figsize=(10,10))
plt.bar(rmse_error_synt.keys(),rmse_error_synt.values())
plt.xticks(rotation='vertical')
plt.ylabel("Root mean square error")

plt.figure(figsize=(10,10))
plt.bar(mape_error_synt.keys(),mape_error_synt.values())
plt.xticks(rotation='vertical')
plt.ylabel("Mean absolute percentage error")

# Train

reglin = linear_model.LinearRegression() 
reglin.fit(Xtrain,Ytrain) 
Y_prediction = reglin.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission0.csv',index=False)

cross_validation_ridge = RidgeCV(alphas = np.arange(0,1,0.01))
cross_validation_ridge.fit(Xtrain,Ytrain)
Y_prediction = cross_validation_ridge.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission1.csv',index=False)

Xtrain_lars_path = np.array(Xtrain,dtype='float64')
Ytrain_lars_path = np.reshape(np.array(Ytrain,dtype='float64'),(Ytrain.shape[0],))
lasso_lars_cv = LassoLarsCV(cv = 4) 
lasso_lars_cv.fit(Xtrain_lars_path,Ytrain_lars_path)
Y_prediction = lasso_lars_cv.predict(Xtest)
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission2.csv',index=False)

spline = s(0)
for i in range(1,39):
    spline += s(i)
gam = LinearGAM(spline).fit(Xtrain,Ytrain)     
# Predict
Y_prediction = gam.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission4.csv',index=False)



