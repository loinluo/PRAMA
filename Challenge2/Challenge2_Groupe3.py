# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

# Chargement des donnes
train_data = pd.read_csv('train_data.csv') 
test_data = pd.read_csv('test_data.csv')

# Supprimer les variables de prix anormales 
def box_plot_outliers(data):
  q1,q3 = data.quantile(.25),data.quantile(.75)
  IQR = q3-q1
  low,up = q1 - 1.5*IQR, q3 + 1.5*IQR
  outlier = data.mask((data<low)|(data>up))
  return outlier
train_data['prix'] = pd.DataFrame(box_plot_outliers(train_data['prix']))
train_data = train_data.drop(train_data[train_data['nb_chambres']>10].index)
train_data = train_data.drop(train_data[train_data['m2_jardin']>10000].index)
train_data.dropna(inplace = True)

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

# classifier 'zipcode' par le prix (10 groupes)
df_zipcode = (train_data.groupby(['zipcode'])['prix'].mean()).to_frame()
df_quantile = df_zipcode.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], numeric_only=True)
dic_zipcode = {}
for i in range(df_zipcode.shape[0]):
    for j in range(10):
        if df_zipcode.iloc[i]['prix'] >= df_quantile.iloc[j]['prix'] and df_zipcode.iloc[i]['prix'] < df_quantile.iloc[j+1]['prix']:
            dic_zipcode[df_zipcode.index[i]] = str(j+1)
        elif df_zipcode.iloc[i]['prix'] == df_quantile.iloc[10]['prix']:
            dic_zipcode[df_zipcode.index[i]] = str(10)

all_features['zipcode'] = all_features['zipcode'].apply(
    lambda x:(dic_zipcode[x]))
    

# One-Hot Encoding pour "zipcode"
all_features = pd.get_dummies(all_features, dummy_na = True)
all_features = all_features.drop(['zipcode_nan'],axis=1)

# Separer les variables de 'train' et 'test'
n_train = train_data.shape[0]
Xtrain = all_features.iloc[:n_train]
Xtest = all_features.iloc[n_train:]

# Obtenir index
numeric_features = Xtrain.dtypes[Xtrain.dtypes != 'object'].index

# Normaliser les donnes par mean et std de train
mean_Xtrain = Xtrain.mean()
std_Xtrain = Xtrain.std()
Xtrain[numeric_features] = Xtrain[numeric_features].apply(
    lambda x:(x - x.mean())/(x.std())) 

for i in range(numeric_features.shape[0]):
    Xtest[numeric_features[i]] = Xtest[numeric_features[i]].apply(
        lambda x:(x - mean_Xtrain[i])/std_Xtrain[i])

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

def get_k_blocs_torch(k, i, X, Y):
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
            X_train = torch.cat([X_train, X_part], 0)
            Y_train = torch.cat([Y_train, Y_part], 0)
    return X_train, Y_train, X_valid, Y_valid 

#Bagging
from sklearn.ensemble import BaggingRegressor
def BR(k, X_train, Y_train, n):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        bagr = BaggingRegressor(n_estimators=n) 
        bagr.fit(data[0],data[1]) 
        Y_prediction = bagr.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result6 = BR(5, Xtrain, Ytrain, 200)
rmse_error_synt['Bagging'] = result6[0]
mape_error_synt['Bagging']  = result6[1]

#predict
bagr = BaggingRegressor(n_estimators=200)
bagr.fit(Xtrain,Ytrain)
Y_prediction = bagr.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission6.csv',index=False)

#CART
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn import tree
treereg  = tree.DecisionTreeRegressor(random_state=0)
treereg.fit(Xtrain,Ytrain)
mse_scorer = make_scorer(mean_squared_error) 
max_depth_CV = treereg.get_depth() 
error = np.zeros(max_depth_CV)
# Calcul de l'erreur pour différentes profondeur de l'arbre
for depth in np.arange(0,max_depth_CV)+1:
    treeCV = tree.DecisionTreeRegressor(max_depth = depth, random_state = 0)
    error[int(depth-1)] = cross_val_score(treeCV,Xtrain,Ytrain,cv=5,scoring=mse_scorer).mean() # cv = 5 <=> validation croisée par k-fold avec k=5
# Choix de la meilleur profondeur, qui minimise l'erreur mse 
best_depth = np.argmin(error)+1
def CART(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        treeregbestCV  = tree.DecisionTreeRegressor(max_depth = best_depth, random_state = 0)
        treeregbestCV.fit(data[0],data[1])  
        Y_prediction = treeregbestCV.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k
result7 = CART(5, Xtrain, Ytrain)
rmse_error_synt['CART'] = result7[0]
mape_error_synt['CART']  = result7[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result7[0],result7[1]))
#predict
cart = tree.DecisionTreeRegressor(max_depth = best_depth, random_state = 0)
cart.fit(Xtrain,Ytrain)
Y_prediction = cart.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission7.csv',index=False)

#Random Forest
from sklearn import ensemble
def RF(k, X_train, Y_train, n):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        randomforestreg = ensemble.RandomForestRegressor(n_estimators=n,random_state=0)
        randomforestreg.fit(data[0],data[1])  
        Y_prediction = randomforestreg.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k


result8 = RF(5, Xtrain, Ytrain, 200)
rmse_error_synt['RF'] = result8[0]
mape_error_synt['RF']  = result8[1]
#predict
rfreg = ensemble.RandomForestRegressor(n_estimators=200,random_state=0)
rfreg.fit(Xtrain,Ytrain)
Y_prediction = rfreg.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission8.csv',index=False)

# GradientBoostingRegressor

gmbreg  = ensemble.GradientBoostingRegressor(random_state=0)
gmbreg.fit(Xtrain,Ytrain)
mse_scorer = make_scorer(mean_squared_error)
max_depth_CV1 = 20
error = np.zeros(max_depth_CV1)
# Calcul de l'erreur pour différentes profondeur de l'arbre
for depth in np.arange(0,max_depth_CV1)+1:
    gmbCV = ensemble.GradientBoostingRegressor(max_depth = depth, random_state = 0)
    error[int(depth-1)] = cross_val_score(gmbCV,Xtrain,Ytrain,cv=5,scoring=mse_scorer).mean() # cv = 5 <=> validation croisée par k-fold avec k=5
# Choix de la meilleur profondeur, qui minimise l'erreur mse
best_depth1 = np.argmin(error)+1

def L2Boosting(k, X_train, Y_train, n_estimators, n_iter_no_change, validation_fraction, learning_rate):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        gmbreg =ensemble.GradientBoostingRegressor(n_estimators=n_estimators,n_iter_no_change=n_iter_no_change,validation_fraction=validation_fraction,learning_rate=learning_rate,max_depth = 8,random_state=0) 
        gmbreg.fit(data[0],data[1])  
        Y_prediction = gmbreg.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k

result9 = L2Boosting(5, Xtrain, Ytrain, 200, 10, 0.1, 0.1)
rmse_error_synt['L2Boosting'] = result9[0]
mape_error_synt['L2Boosting']  = result9[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result9[0],result9[1]))

#predict 选取最优参数
gmbreg = ensemble.GradientBoostingRegressor(n_estimators=200, n_iter_no_change=50, validation_fraction=0.1, learning_rate=0.1, max_depth = 8,random_state=0)
gmbreg.fit(Xtrain,Ytrain)
Y_prediction = gmbreg.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission9.csv',index=False)

#SVM（SVR）
from sklearn.svm import SVR
def SVM(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        boston_svr = SVR(kernel='rbf', gamma='auto', C=100000, epsilon = 0.00001) 
        boston_svr.fit(data[0],data[1])
        Y_prediction = boston_svr.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
    return rmse_sum / k, mape_sum / k
result10 = SVM(5, Xtrain, Ytrain)
rmse_error_synt['SVM'] = result10[0]
mape_error_synt['SVM']  = result10[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result10[0],result10[1]))
#predict
svr = SVR(kernel='rbf', gamma='auto', C=100000, epsilon = 0.00001)
svr.fit(Xtrain,Ytrain)
Y_prediction = svr.predict(Xtest)
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission10.csv',index=False)

#NN
train_features = torch.tensor(Xtrain.values, dtype=torch.float32)
train_labels = torch.tensor(Ytrain.values, dtype=torch.float32)
test_features = torch.tensor(Xtest.values, dtype=torch.float32)

loss = nn.MSELoss()

def get_net(train_features):
    net = nn.Sequential(
          nn.Linear(train_features.shape[1], 128),
          nn.ReLU(),
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 1)
        )
    return net


def nn_train(net,train_features, train_labels, test_features, test_labels,
             num_epochs, learning_rate, weight_decay, batch_size):
    train_rmse, test_rmse, test_mape = [], [], []
    train_iter = d2l.load_array((train_features, train_labels.unsqueeze(-1)),batch_size)
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,
                                 weight_decay=weight_decay) 
    train_labels = train_labels.detach().numpy()
    train_labels = train_labels.reshape(len(train_labels))
    test_labels = test_labels.detach().numpy()
    test_labels = test_labels.reshape(len(test_labels)) 
    for epoch in range(num_epochs): 
        for X, Y in train_iter:
            optimizer.zero_grad() 
            l = loss(net(X), Y) 
            l.backward() 
            optimizer.step() 
        Y_train = net(train_features)
        Y_train = Y_train.detach().numpy()
        Y_train = Y_train.reshape(len(Y_train))
        train_rmse.append(rmse(train_labels, Y_train))     
        Y_prediction = net(test_features)
        Y_prediction = Y_prediction.detach().numpy()
        Y_prediction = Y_prediction.reshape(len(Y_prediction))
        test_rmse.append(rmse(test_labels, Y_prediction))
        test_mape.append(mape(test_labels, Y_prediction))
        
    return train_rmse, test_rmse, test_mape 
        
def nn_cv(k, X_train, Y_train, num_epochs, learning_rate, weight_decay, batch_size):
    rmse_train, rmse_sum, mape_sum = 0, 0, 0
    for i in range(k):
        data = get_k_blocs_torch(k, i, X_train, Y_train)
        net = get_net(data[0]) 
        train_rmse, test_rmse, test_mape = nn_train(net,*data, num_epochs, learning_rate,
                                        weight_decay, batch_size)
        rmse_train += train_rmse[-1] 
        rmse_sum += test_rmse[-1] 
        mape_sum += test_mape[-1]
        if i == 0:
            d2l.plot(list(range(1,num_epochs + 1)),[train_rmse,test_rmse], 
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid']) 
            
    return rmse_sum / k, mape_sum / k


k, num_epochs, learning_rate, weight_decay, batch_size = 5, 100, 0.1, 0, 256
result11 = nn_cv(k, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size)
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result11[0],result11[1]))
rmse_error_synt['BPNN'] = result11[0]
mape_error_synt['BPNN']  = result11[1]

#predict
net = get_net(test_features)
train_iter = d2l.load_array((train_features, train_labels.unsqueeze(-1)),batch_size)
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,
                                 weight_decay=weight_decay)
for epoch in range(num_epochs):
    for X, Y in train_iter:
        optimizer.zero_grad()
        l = loss(net(X), Y)
        l.backward()
        optimizer.step()
    Y_prediction = net(test_features)

Y_prediction = Y_prediction.detach().numpy()
Y_prediction = Y_prediction.reshape(len(Y_prediction))
test_data['prix'] = Y_prediction
submission = pd.concat([test_data['id'],test_data['prix']], axis=1)
submission.to_csv('submission11.csv',index=False)

plt.figure(figsize=(10,10))
plt.bar(rmse_error_synt.keys(),rmse_error_synt.values())
plt.xticks(rotation='vertical')
plt.ylabel("Root mean square error")

plt.figure(figsize=(10,10))
plt.bar(mape_error_synt.keys(),mape_error_synt.values())
plt.xticks(rotation='vertical')
plt.ylabel("Mean absolute percentage error")