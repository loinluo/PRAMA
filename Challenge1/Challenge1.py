from sklearn import linear_model
from statsmodels.nonparametric import kernel_regression
import pygam
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#Chargement des donnes
train_data = pd.read_csv("train_data.csv") 
test_data = pd.read_csv("test_data.csv")

#Création d'un échantillon apprentissage et test
# from sklearn.model_selection import train_test_split
Xtrain = train_data.drop(['id','prix'],axis=1)
Ytrain = train_data['prix']
Xtest = test_data.drop(['id'],axis=1)
# One-Hot Encoding pour "date"
Xtrain = pd.get_dummies(Xtrain, dummy_na = True)
Xtest = pd.get_dummies(Xtest, dummy_na = True)
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,train_size=0.75,random_state=111) 

#Définition des risques empiriques
from sklearn.metrics import mean_squared_error
# Definition of the root mean squared error
def rmse(y,yprime):
    return np.sqrt(mean_squared_error(y,yprime))
# Definition of the mean absolute percentage error
def mape(y,yprime):
    return 100 * np.sum(np.abs(y-yprime)/y)/len(y)
mape_error_synt = {}
rmse_error_synt = {}

# from pandas.plotting import scatter_matrix
# train_data.describe()

# import warnings
# warnings.filterwarnings('ignore')
# scatter_matrix(train_data, diagonal="kde",figsize=(25,25))
# pass

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
        return rmse_sum / k, mape_sum / k, data[3], Y_prediction

result1 = reglin(5, Xtrain, Ytrain)
rmse_error_synt['Multiple linear regression'] = result1[0]
mape_error_synt['Multiple linear regression']  = result1[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result1[0],result1[1]))

#Régression ridge
from sklearn.linear_model import RidgeCV,Ridge

def ridge(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        cross_validation_ridge = RidgeCV(alphas = np.arange(10,4000,1)+0.1)
        cross_validation_ridge.fit(data[0],data[1])
        Y_prediction = cross_validation_ridge.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
        return rmse_sum / k, mape_sum / k, data[3], Y_prediction
    
result2 = ridge(5, Xtrain, Ytrain)
rmse_error_synt['Ridge regression'] = result2[0]
mape_error_synt['Ridge regression']  = result2[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result2[0],result2[1]))

#Régression LASSO-LARS
from sklearn.linear_model import lars_path
from sklearn.linear_model import LassoLarsCV

def lassolars(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        Xtrain_lars_path = np.array(data[0],dtype='float64')
        Ytrain_lars_path = np.reshape(np.array(data[1],dtype='float64'),(17147,))
        # path  = lars_path(Xtrain_lars_path,Ytrain_lars_path,method='lasso')
        # xx = np.sum(np.abs(path[2].T), axis=1)
        # xx /= xx[-1]
        # plt.figure(figsize=(20,20))
        # plt.plot(xx, path[2].T)
        # ymin, ymax = plt.ylim()
        # plt.vlines(xx, ymin, ymax, linestyle='dashed')        
        lasso_lars_cv = LassoLarsCV(cv = 4) # Cross-validtion using k-folds with k=5
        lasso_lars_cv.fit(Xtrain_lars_path,Ytrain_lars_path)
        Y_prediction = lasso_lars_cv.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
        return rmse_sum / k, mape_sum / k, data[3], Y_prediction

result3 = lassolars(5, Xtrain, Ytrain)
rmse_error_synt['LASSO-LARS regression'] = result3[0]
mape_error_synt['LASSO-LARS regression']  = result3[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result3[0],result3[1]))

#Méthode de Nadaraya-Watson
def NW(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        kreg = kernel_regression.KernelReg(data[1],data[0]['m2_interieur'],"c")
        Y_prediction = kreg.fit(data[2]['m2_interieur'])[0]
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
        return rmse_sum / k, mape_sum / k, data[3], Y_prediction

result4 = NW(5, Xtrain, Ytrain)
rmse_error_synt['N-W kernel regression'] = result4[0]
mape_error_synt['N-W kernel regression']  = result4[1]
print("Moyenne de RMSE={} \nMoyenne de MAPE={}".format(result4[0],result4[1]))

# Generalized Additive Models
from pygam import LinearGAM, s, f
def GAM(k, X_train, Y_train):
    rmse_sum, mape_sum = 0, 0
    for i in range(k):
        data = get_k_blocs_data(k, i, X_train, Y_train)
        gam = LinearGAM( s(3) + s(4)+s(5)+s(6)+s(7)) .fit(Xtrain,Ytrain)
        Y_prediction = gam.predict(data[2])
        rmse_sum += rmse(data[3],Y_prediction)
        mape_sum += mape(data[3],Y_prediction)
        return rmse_sum / k, mape_sum / k, data[3], Y_prediction

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


