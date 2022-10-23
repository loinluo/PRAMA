# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:37:48 2022

@author: Lenovo
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import warnings
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification

# 数据预处理部分：

fundamentals_df = pd.read_csv('fundamentals.csv')
securities_df = pd.read_csv('securities.csv')



sector_list = securities_df['GICS Sector'].unique() # 提取出所有的sector的名称

sector_list_num = np.arange(len(sector_list)) 
sector_dict = dict(zip(sector_list,sector_list_num)) # sector表（给sector赋标签值）

securities_df_o = securities_df.copy(deep = True)
securities_df_o['sector_label'] = securities_df_o['GICS Sector'].map(lambda x:sector_dict[x]) # 添加sector的数值标签

securities_group = securities_df_o.loc[:,['Ticker symbol','sector_label']] # 提取股票代码和sector标签

securities_group_dict = securities_group.set_index(['Ticker symbol'])['sector_label'].to_dict() # 建立股票代码对应的sector表

fundamentals_df_o = fundamentals_df.copy(deep = True)
fundamentals_df_o['sector_label'] = fundamentals_df_o['Ticker Symbol'].map(lambda x:securities_group_dict[x]) # 根据股票名称添加sector标签

fundamentals_df_o = fundamentals_df_o.drop(['For Year','Unnamed: 0','Period Ending'],axis = 1) # 去除无意义的列以及与时间相关的列
fundamentals_df_o = fundamentals_df_o.dropna() # 去除NA

# 根据securities得到股票代码对应的行业，寻找fundamentals中的基础变量与行业的关系，进行分类

# numeric_features = fundamentals_df_o.dtypes[fundamentals_df_o.dtypes == 'float64'].index # 浮点型变量的索引（除label之外）

# Xtrain = fundamentals_df_o.drop(['sector_label'],axis = 1) # 除行业以外的变量
# Ytrain = fundamentals_df_o['sector_label'] # 行业

# # Normaliser les donnes par mean et std de train，归一化

# mean_Xtrain = Xtrain.mean()
# std_Xtrain = Xtrain.std()
# Xtrain[numeric_features] = Xtrain[numeric_features].apply(
#     lambda x:(x - x.mean())/(x.std())) 

# for i in range(len(numeric_features)):
#     Xtrain[numeric_features[i]] = Xtrain[numeric_features[i]].apply(
#         lambda x:(x - mean_Xtrain[numeric_features[i]])/std_Xtrain[numeric_features[i]])


# # K-blocs CV
# # Diviser les donnes
# def get_k_blocs_data(k, i, X, Y):
#     assert k > 1
#     blocs_size = X.shape[0] // k
#     X_train, Y_train = None, None
#     for j in range(k):
#         idx = slice(j * blocs_size, (j + 1) * blocs_size)
#         X_part, Y_part = X[idx], Y[idx]
#         if j == i:
#             X_valid, Y_valid = X_part, Y_part
#         elif X_train is None:
#             X_train, Y_train = X_part, Y_part
#         else:
#             X_train = pd.concat([X_train, X_part])
#             Y_train = pd.concat([Y_train, Y_part])
#     return X_train, Y_train, X_valid, Y_valid


# # 基础指标的协方差
# fig=plt.figure(figsize=(24,16),dpi=80)
# sns.heatmap(fundamentals_df_o.corr(),annot =True,vmin = 0, vmax = 1)

# # 利用PCA将变量减为2个
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# pca.fit(Xtrain)
# X_t_train = pca.transform(Xtrain)
# X_t_train = pd.DataFrame(X_t_train)

# from sklearn.svm import SVC

# def SVM(k, X_train, Y_train):
#     score_train_sum, score_test_sum = 0, 0
#     for i in range(k):
#         data = get_k_blocs_data(k, i, X_train, Y_train)
#         clf = SVC(kernel='rbf', gamma=100, C=0.3, decision_function_shape='ovr') 
#         clf.fit(data[0],data[1])
#         score_train_sum += clf.score(data[0], data[1])
#         score_test_sum += clf.score(data[2], data[3])
#     return score_train_sum / k, score_test_sum / k
# result = SVM(5, X_t_train, Ytrain)
# print(result[1])

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import seaborn as sns
import random
from datetime import datetime
import matplotlib.ticker as ticker

prices_df = pd.read_csv('prices-split-adjusted.csv')

# choisir un secteur (Consumer Discretionary) 
fundamentals_selected = fundamentals_df_o.loc[fundamentals_df_o['sector_label']==3,:] # 选取label为3的行业
fundamentals_selected = fundamentals_selected.reset_index(drop = True) # 重新生成索引
symbol_selected = fundamentals_selected.iloc[fundamentals_selected['Total Assets'].idxmax(),:]['Ticker Symbol'] # 找到总资产最高的股票为'F'(Ford)

# 提取福特的价格数据
price_selected = prices_df.loc[prices_df['symbol']==symbol_selected,:] # 提取出price文件中的股票代码为'F'的数据
price_selected = price_selected.reset_index(drop = True) # 重新生成索引
series_selected = price_selected['close'] # 每日的收盘价close（需要预测的变量）

# 股价走势图（2010~2016）
plt.figure(figsize=(36,8),dpi=80)
date = price_selected.loc[:,['date']]
date = list(date['date'])
price = price_selected.loc[:,['close']]
plt.plot_date(date,price,fmt='b.',linestyle='solid',label = 'close_prices' )
plt.title('Close price',fontsize=15)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.xticks(rotation = 90,ha = 'center')

# 铁矿石价格处理
iron_price = pd.read_csv('z_iron_price.csv')
iron_price = iron_price.rename(columns={'Float':'time_float'}) # 将Float列重命名为time_float
date_list = pd.read_csv('z_date_list.csv')
iron_ore_price =  pd.merge(date_list, iron_price, how='left', on=['time_float']) # 将钢铁的price和date依据共同列time_float组合起来
iron_ore_price_temp = iron_ore_price.copy(deep = True)

# 由于只有每个月1日的价格，需要补充缺失值（直接插值，每个月内的价格都看做从月初到下月初的直线）
flag = 0
for index in iron_ore_price_temp.index:

  if (not pd.isnull(iron_ore_price_temp.iloc[index]['Price']))and(flag == 0):
    flag = 1
    index_temp = []
    start_price = iron_ore_price_temp.iloc[index]['Price']
    continue

  if (not pd.isnull(iron_ore_price_temp.iloc[index]['Price']))and(flag == 1):

    end_price = iron_ore_price_temp.iloc[index]['Price']
    price_fill = np.arange(start_price,end_price,(end_price - start_price)/len(index_temp))
    for i in index_temp:
      iron_ore_price_temp.loc[i,['Price']] = price_fill[index_temp.index(i)]
    flag = 0
    if (not pd.isnull(iron_ore_price_temp.iloc[index]['Price']))and(flag == 0):
      flag = 1
      index_temp = []
      start_price = iron_ore_price_temp.iloc[index]['Price']
      continue
    continue
  if flag == 1:
    index_temp.append(index)

iron_price_list = iron_ore_price_temp.drop(['time_date','Date'],axis = 1)
iron_price_list = iron_price_list.dropna()
iron_price_list = iron_price_list.rename(columns = {'time_float':'float'})
price_co = pd.merge(price_selected,iron_price_list,how = 'left', on = ['float'])

plt.figure(figsize=(24,8),dpi=80)
date = price_co.loc[:,['date']]
date = list(date['date'])
price = price_co.loc[:,['close']]
iron_price = price_co.loc[:,['Price']]
plt.plot_date(date,price,fmt='r',linestyle='solid',label = 'close_price' )
plt.plot_date(date,iron_price,fmt='b.',linestyle='solid',label = 'iron_price' )
plt.legend(labels=['Ford price','Iron price'],loc='best',fontsize = 15)
plt.title('Ford price and Iron price',fontsize=15)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.xticks(rotation = 90,ha = 'center')

#加入铁矿石价格协变量的预测
price_data = price_co.loc[:,['date','close','Price']]
price_data_used = price_data.loc[:,['close','Price']]
# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a univariate dataset into train/test sets
def split_dataset(data): # 将数据集后30个设置为测试集，前面的都是训练集
	# split into standard weeks
	train, test = data[0:-30], data[-30:]
	# restructure into windows of weekly data
	train = np.array(np.split(train, len(train)/1))
	test = np.array(np.split(test, len(test)/1))
	return train, test
 
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
	# flatten data
	print(train.shape)
	if len(train.shape) == 2:

		data = train.reshape((train.shape[0],train.shape[1] ))
	else:
		data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)
 
# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 1, 30, 20
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
 
# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	
	if len(data.shape) == 2:
		data = data.reshape((data.shape[0],data.shape[1]))
	else:
		data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
 
# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	# if len(test.shape):
	# 	score, scores = evaluate_forecasts(test[:,  0], predictions)
	# else:
	# 	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	score = 0
	scores = 0

	return score, scores, predictions

train,test = split_dataset(price_data_used.values)

train_x, train_y = to_supervised(train, 1)

n_input = 1

score, scores, predictions = evaluate_model(train, test, n_input)

predictions = predictions.reshape(len(predictions))

pre = predictions
true_value = price_co[-30:].loc[:,'close']
true_value = list(true_value)

pyplot.plot(true_value)
pyplot.plot(pre)
pyplot.show()

# 训练与预测部分：

import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from pandas import read_table
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from pyhht.visualization import plot_imfs
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

values = series_selected

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

class PrequentialSplit(_BaseKFold): # 预先分割类（分滑动时间窗）
    STRATEGY_PREQ_BLS = 'preq-bls'
    STRATEGY_PREQ_SLID_BLS = 'preq-slid-bls'
    STRATEGY_PREQ_BLS_GAP = 'preq-bls-gap'

    def __init__(self, strategy='preq-bls', base_size=None, n_splits=5, stride=1, *, max_train_size=None,
                 test_size=None, gap_size=0):
        super(PrequentialSplit, self).__init__(n_splits=max(n_splits, 2), shuffle=False, random_state=None)

        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.base_size = base_size
        self.stride = stride
        self.n_folds = n_splits
        self.strategy = strategy
        self.fold_size = None

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1 # 折数等于n_split+1
        gap_size = self.gap_size

        base = 0
        if self.base_size is not None and self.base_size > 0:
            base = self.base_size
        base += n_samples % n_folds  # base为样本个数除以折数的余数

        if self.test_size is not None and self.test_size > 0:
            test_size = self.test_size
        else:
            test_size = (n_samples - base) // n_folds 
        self.test_size = test_size

        if self.n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n_samples))

        first_test = n_samples - test_size*n_splits # first_test等于样本个数-测试集大小*分割个数
        if first_test < 0:
            raise ValueError(
                ("Too many splits={0} for number of samples"
                 "={1} with test_size={2}").format(n_splits, n_samples, test_size))
        indices = np.arange(n_samples)
        if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
            test_starts = range(first_test * 2 + base, n_samples, test_size)
        else:
            test_starts = range(first_test + base, n_samples, test_size) # 测试集开始元素为first_test + base到样本个数，步长为测试集大小的一个序列
        last_step = -1
        for fold, test_start in enumerate(test_starts): # fold为索引，test_start为元素
            if last_step == fold // self.stride:
                # skip this fold
                continue
            else:
                last_step = fold // self.stride # 最后一步为测试集开始元素整除stride
            if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS:
                train_end = test_start - gap_size
                if self.max_train_size and self.max_train_size < train_end:
                    yield (indices[train_end - self.max_train_size:train_end],
                           indices[test_start:test_start + test_size])
                else:
                    yield (indices[:max(train_end, 0)],
                           indices[test_start:test_start + test_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_SLID_BLS: # 我们选用这种策略
                if self.max_train_size and self.max_train_size < test_start:
                    yield (indices[test_start - self.max_train_size:test_start], # 训练集索引
                           indices[test_start:test_start + test_size])
                else:
                    yield (indices[test_start - (test_size + base):test_start], 
                           indices[test_start:test_start + test_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
                yield (indices[:test_start - test_size], indices[test_start:test_start + test_size])
            else:
                raise ValueError(f'{self.strategy} is not supported')

def difference(dataset, interval=1): # 返回数据集以1为间隔的一阶差分（近似每个点的一阶导）
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def timeseries_to_supervised(data, lag=1): # 将时间后移一天
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def inverse_difference(history, yhat, interval=1): # 逆差
    return yhat + history[-interval] 

def scale(train, test): # 将数据集缩放到-1到1之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value): # 
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    # print(array)
    # b = numpy.array([value])
    # print(b)
    # array = np.column_stack((array[0][0],b))
    # print(array)
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# def invert_scale(scaler, X, value):
#     new_row = [x for x in X] + [value]
#     array = numpy.array(new_row)
#     array = array.reshape(1, len(array))
#     inverted = scaler.inverse_transform(array)
#     return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons,node): # lstm模型,参数为：训练集，批尺寸，传播次数，神经元，节点
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(node))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forcast_lstm(model, batch_size, X): # lstm预测
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

def box_plot_outliers(data): # 去除异常值
  q1,q3 = data.quantile(.25),data.quantile(.75)
  IQR = q3-q1
  low,up = q1 - 1.5*IQR, q3 + 1.5*IQR
  outlier = data.mask((data<low)|(data>up))
  return outlier

def lstm_pre_eemd_valide(imfsValues,RESIDUE,epochs,neurons,node,Ytrain,size = 30): 

  imfsValues_pre = []
  for imf in imfsValues:
    raw_values = imf
    diff_values = difference(raw_values,1) 
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    train, test = supervised_values[0:-size], supervised_values[-size:]
    scaler, train_scaled, test_scaled = scale(train, test)

    lstm_model = fit_lstm(train_scaled, 1, epochs, neurons,node)

    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1) #用模型对训练数据矩阵进行预测
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forcast_lstm(lstm_model, 1, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i+1]
    
    imfsValues_pre.append(predictions)
  
    imfsValues_pre = pd.DataFrame(imfsValues_pre)
    imfsValues_pre_test = imfsValues_pre.transpose()
    imfsValues_pre_test['residue'] = RESIDUE[-size:]
    imfsValues_pre_test['sum'] = imfsValues_pre_test.apply(lambda x:x.sum(),axis = 1)
    pre = list(np.array(imfsValues_pre_test['sum']))
    print(pre)
    mape = mean_absolute_percentage_error(Ytrain, pre)
    print(mape)  
    return imfsValues_pre_test, mape

# imfsValues_list,residue_list,Ytrain = validation(series_selected,len(series_selected),k=5,n_splits=5)

# Ytrain_temp = np.array(Ytrain[0])
# pre_temp,rmse_temp = lstm_pre_eemd_valide(imfsValues_list[0],residue_list[0],20,20,30,Ytrain_temp,size = len(Ytrain_temp))

# pre = []
# rmse = []
# for i in range(len(imfsValues_list)):
#   Ytrain_temp = np.array(Ytrain[i])
#   print(Ytrain_temp)
#   pre_temp,rmse_temp = lstm_pre_eemd_valide(imfsValues_list[i],residue_list[i],10,20,30,Ytrain_temp,size = len(Ytrain_temp))
#   pre.append(pre_temp)
#   rmse.append(rmse_temp)
  

df_epalia_c_d_1_index = np.array(series_selected.index) # 收盘价格的索引转为数组
df_epalia_c_d_1_value = np.array(series_selected)  # 收盘价格转为数组
# rmse = sqrt(mean_squared_error(df_epalia_c_d_1_value[-30:], pre)) 


from PyEMD import EEMD

# 将收盘价格close信号进行EMD（经验模态分解），成为有限个本征模函数IMF和一个残差Residue
eemd = EEMD()
eemd.trials = 50 # 试验trial的次数
eemd.noise_seed(12345) # 噪声种子
EEMD_IMFS = eemd.eemd(df_epalia_c_d_1_value,df_epalia_c_d_1_index) # 将收盘价格信号分解为本征模函数IMF（Intrinsic Mode Function)
EEMD_num_IMFS = EEMD_IMFS.shape[0] # IMF个数

# 可视化展示原信号和IMF
c = int(np.floor(np.sqrt(EEMD_num_IMFS + 1))) # 图片列数 （一共10个图像，开方向下取整）
r = int(np.ceil((EEMD_num_IMFS + 1) / c)) # 图片行数（图像数除以列数向上取整）

plt.figure(figsize=(24,15),dpi=80)
plt.ioff()
plt.subplot(r, c, 1)
plt.plot(df_epalia_c_d_1_index, df_epalia_c_d_1_value, 'r')
plt.xlim((df_epalia_c_d_1_index.min(), df_epalia_c_d_1_index.max()))
plt.title("Original signal")

for num in range(EEMD_num_IMFS):
    plt.subplot(r, c, num + 2)
    plt.plot(df_epalia_c_d_1_index, EEMD_IMFS[num], 'g')
    plt.xlim((df_epalia_c_d_1_index.min(), df_epalia_c_d_1_index.max()))
    plt.title("Imf " + str(num + 1))

plt.show() 

EEMD_IMFS,RESIDUE = eemd.get_imfs_and_residue() # 获取IMFs和Residue
imfsValues = EEMD_IMFS # imf的值，本质是一个二维数组，每行的数据为一个imf信号


n_splits = 5
valid = PrequentialSplit('preq-slid-bls', n_splits=n_splits, max_train_size=365, test_size=30) # 定义一个分5块，最大训练集大小为365个，测试集大小为30的预先分割类
Time_splited_group = valid.split(values) # 将收盘价数据集values进行分割（5组，训练集365个，测试集30个）
# 收集训练集和测试集的索引
train_index_group = [] 
test_index_group = []
for train_index,test_index in Time_splited_group:
  train_index_group.append(train_index)
  test_index_group.append(test_index) 

# 将训练集和测试集的索引合并到一起， 变为5*(365+30)的list
train_c_index_group = []
for i in range(len(train_index_group)):
  train_temp = list(train_index_group[i].reshape(len(train_index_group[i])))
  test_temp = list(test_index_group[i].reshape(len(test_index_group[i])))
  train_c_index_group.append(train_temp+(test_temp))

# 将每组训练集和测试集的九个imf和residue整合，得到一个5*9*395的imf和一个5*1*395的residue
imfsValues_group = []
residue_group = []
for i in range(len(train_c_index_group)):
  imfsValues_temp = []
  residue_temp = []
  for imf in imfsValues:
    imfsValues_temp.append(imf[train_c_index_group[i]])
  imfsValues_group.append(imfsValues_temp)
  residue_group.append(RESIDUE[train_c_index_group[i]])


# 选择参数函数，返回一组参数下的rmse均值相反数(因为我们找的是最小值，而贝叶斯优化只能找最大值)  
def para_lstm(values,train_c_index_group,test_index_group,imfsValues_group,residue_group,epochs,neurons,node):
  pre_total = []
  mape_total = []
  for i in range(len(train_c_index_group)):
    size = len(test_index_group[i])
    Ytrain = values[train_c_index_group[i][-size:]]
    pre,mape = lstm_pre_eemd_valide(imfsValues_group[i],residue_group[i],epochs,neurons,node,Ytrain,size = size)
    pre_total.append(pre)
    mape_total.append(mape)
  return -np.array(mape_total).mean()


# warnings.filterwarnings("ignore")

# 调参

# X, y = make_classification(n_samples=10000,n_features=20,n_classes=2,random_state=2) # 随机生成一组X和y（20个特征的0,1分类）
# data = lgb.Dataset(X,y)

# 将参数取整（贝叶斯优化器随机取的数不是整数，需要变为整数）
def lgb_cv(epochs,neurons,node):
        params = {'objective':'binary'}
        params['epochs'] = int(round(epochs))# round是将数字四舍五入到最接近的整数
        params['neurons'] = int(round(neurons))
        params["node"] = int(round(node))

        mape = para_lstm(values,train_c_index_group,test_index_group,imfsValues_group,residue_group,params['epochs'],params['neurons'],params["node"])

        return mape

# 贝叶斯优化器，找到函数的最大值，优化三个参数
lgb_bo = BayesianOptimization(  
        lgb_cv,
        {'epochs': (5, 30),
        'neurons': (5, 40),
        'node': (5,50)}
    )        

lgb_bo.maximize(init_points=5,n_iter=5) #init_points表示初始点，n_iter代表迭代次数（即采样数） 乘积为迭代总数
print (lgb_bo.max)


# para_grid = {'epochs': [10],
#              'neurons':[ 20],
#              'node':[  30, 40]}
# combinations = list(ParameterGrid(para_grid))



def lstm_pre_eemd(imfsValues,RESIDUE,epochs,neurons,node):
  imfsValues_pre = []
  for imf in imfsValues:
    raw_values = imf
    diff_values = difference(raw_values,1) 
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    train, test = supervised_values[0:-30], supervised_values[-30:]
    scaler, train_scaled, test_scaled = scale(train, test)

    lstm_model = fit_lstm(train_scaled, 1, epochs, neurons,node)

    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    print(lstm_model.predict(train_reshaped, batch_size=1))
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forcast_lstm(lstm_model, 1, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i+1]
    imfsValues_pre.append(predictions)
  
  imfsValues_pre = pd.DataFrame(imfsValues_pre)
  imfsValues_pre_test = imfsValues_pre.transpose()
  imfsValues_pre_test['residue'] = RESIDUE[-30:]
  imfsValues_pre_test['sum'] = imfsValues_pre_test.apply(lambda x:x.sum(),axis = 1)
  pre = np.array(imfsValues_pre_test['sum'])
  mape = mean_absolute_percentage_error(df_epalia_c_d_1_value[-30:], pre)  
  return imfsValues_pre_test, mape

# rmse = []
# for param in combinations:
#   pre,rmse_temp = lstm_pre_eemd(imfsValues,RESIDUE,param['epochs'],param['neurons'],param['node'])
#   rmse.append(rmse_temp)
#   print(rmse)
  
# combinations[np.argmin(rmse)]

# param = combinations[np.argmin(rmse)]

# imfsValues_pre_final, rmse_final = lstm_pre_eemd(imfsValues,RESIDUE,param['epochs'],param['neurons'],param['node'])

imfsValues_pre_final, rmse_final = lstm_pre_eemd(imfsValues,RESIDUE,13,21,17)
pre = np.array(imfsValues_pre_final['sum'])
mape = mean_absolute_percentage_error(df_epalia_c_d_1_value[-30:], pre)
mape_percent = 100*mape
print('Test MAPE:%.3f' % mape_percent)
true_value = df_epalia_c_d_1_value[-30:]

pyplot.plot(true_value)
pyplot.plot(pre)
pyplot.show()

plt.figure(figsize=(24,8),dpi=80)
date = price_selected.loc[1000:,['date']]
date = list(date['date'])
date_pre = price_selected.loc[price_selected.tail(30).index,['date']]
date_pre = list(date_pre['date'])
collect = price_selected.loc[1000:,['close']]
# collect_pre = df_epalia_c_d_1.loc[train_index_group[i],['Nb de palettes totales']]
# collect_pre = list(collect_pre['Nb de palettes totales'])
collect_pre = pre
collect_pre = list(collect_pre)

# pre_temp = collect_pre.extend(pre_total[i])
plt.plot_date(date,collect,fmt='b.',linestyle='solid',label = 'real_close_price' )
plt.plot_date(date_pre,collect_pre,fmt='b.',color='r',linestyle='solid',label = 'pre_close_price')
plt.legend(labels=['real_close_price','pre_close_price'],loc='best',fontsize = 15)

plt.title('Real Close Prices and Pre Close Prices',fontsize=15)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(8))
plt.xticks(rotation = 90,ha = 'center')

############################################################




price_iron = price_data_used['Price']

def lstm_pre_eemd_valide_2(imfsValues,RESIDUE,epochs,neurons,node,Ytrain,size = 30,j=1): 

  imfsValues_pre = []
  for imf in imfsValues:
    raw_values = imf
    diff_values = difference(raw_values,1)
    price_iron_v = price_iron[train_c_index_group[j]].reset_index(drop = True)
    df = concat([diff_values,price_iron_v[1:].reset_index(drop = True)],axis=1)
    supervised = timeseries_to_supervised(df, 1)
    supervised = supervised.iloc[:,0:3]
    supervised_values = supervised.values
    train, test = supervised_values[0:-size], supervised_values[-size:]
    scaler, train_scaled, test_scaled = scale(train, test)

    lstm_model = fit_lstm(train_scaled, 1, epochs, neurons,node)

    train_reshaped = train_scaled[:, 0:2].reshape(len(train_scaled), 1, 2)
    lstm_model.predict(train_reshaped, batch_size=1) #用模型对训练数据矩阵进行预测
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forcast_lstm(lstm_model, 1, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)
    
    imfsValues_pre.append(predictions)
  
  imfsValues_pre = pd.DataFrame(imfsValues_pre)
  imfsValues_pre_test = imfsValues_pre.transpose()
  imfsValues_pre_test['residue'] = RESIDUE[-size:]
  imfsValues_pre_test['sum'] = imfsValues_pre_test.apply(lambda x:x.sum(),axis = 1)
  pre = list(np.array(imfsValues_pre_test['sum']))
  mape = mean_absolute_percentage_error(Ytrain, pre)
  return imfsValues_pre_test, mape

def para_lstm_2(values,train_c_index_group,test_index_group,imfsValues_group,residue_group,epochs,neurons,node):
  pre_total = []
  mape_total = []
  for i in range(len(train_c_index_group)):
    size = len(test_index_group[i])
    Ytrain = values[train_c_index_group[i][-size:]]    
    pre,mape = lstm_pre_eemd_valide_2(imfsValues_group[i],residue_group[i],epochs,neurons,node,Ytrain,size = size,j=i)
    pre_total.append(pre)
    mape_total.append(mape)
  return -np.array(mape_total).mean()


def lgb_cv_2(epochs,neurons,node):
        params = {'objective':'binary'}
        params['epochs'] = int(round(epochs))# round是将数字四舍五入到最接近的整数
        params['neurons'] = int(round(neurons))
        params["node"] = int(round(node))

        mape = para_lstm_2(values,train_c_index_group,test_index_group,imfsValues_group,residue_group,params['epochs'],params['neurons'],params["node"])

        return mape


n_splits = 5
valid = PrequentialSplit('preq-slid-bls', n_splits=n_splits, max_train_size=365, test_size=30) # 定义一个分5块，最大训练集大小为365个，测试集大小为30的预先分割类
Time_splited_group = valid.split(values) # 将收盘价数据集values进行分割（5组，训练集365个，测试集30个）
# 收集训练集和测试集的索引
train_index_group = [] 
test_index_group = []
for train_index,test_index in Time_splited_group:
  train_index_group.append(train_index)
  test_index_group.append(test_index) 

# 将训练集和测试集的索引合并到一起， 变为5*(365+30)的list
train_c_index_group = []
for i in range(len(train_index_group)):
  train_temp = list(train_index_group[i].reshape(len(train_index_group[i])))
  test_temp = list(test_index_group[i].reshape(len(test_index_group[i])))
  train_c_index_group.append(train_temp+(test_temp))

price_iron = price_iron[0:-1]


# 将每组训练集和测试集的九个imf和residue整合，得到一个5*9*395的imf和一个5*1*395的residue
imfsValues_group = []
residue_group = []
for i in range(len(train_c_index_group)):
  imfsValues_temp = []
  residue_temp = []
  for imf in imfsValues:
    imfsValues_temp.append(imf[train_c_index_group[i]])
  imfsValues_group.append(imfsValues_temp)
  residue_group.append(RESIDUE[train_c_index_group[i]])

# 贝叶斯优化器，找到函数的最大值，优化三个参数
lgb_bo = BayesianOptimization(  
        lgb_cv_2,
        {'epochs': (5, 30),
        'neurons': (5, 40),
        'node': (5,50)}
    )        

lgb_bo.maximize(init_points=5,n_iter=5) #init_points表示初始点，n_iter代表迭代次数（即采样数） 乘积为迭代总数
print (lgb_bo.max)

raw_values = imfsValues[0]
diff_values = difference(raw_values,1)
df = concat([diff_values,price_iron[1:].reset_index(drop = True)],axis=1)
supervised = timeseries_to_supervised(df, 1)
supervised = supervised.iloc[:,0:3]
supervised = supervised[:-1]


def lstm_pre_eemd_2(imfsValues,RESIDUE,epochs,neurons,node):
  imfsValues_pre = []
  for imf in imfsValues:
    raw_values = imf
    diff_values = difference(raw_values,1)
    df = concat([diff_values,price_iron[1:].reset_index(drop = True)],axis=1)
    supervised = timeseries_to_supervised(df, 1)
    supervised = supervised.iloc[:,0:3]
    supervised_values = supervised.values
    train, test = supervised_values[0:-30], supervised_values[-30:]
    scaler, train_scaled, test_scaled = scale(train, test)

    lstm_model = fit_lstm(train_scaled, 1, epochs, neurons,node)

    train_reshaped = train_scaled[:, 0:2].reshape(len(train_scaled), 1, 2)
    print(lstm_model.predict(train_reshaped, batch_size=1))
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forcast_lstm(lstm_model, 1, X)
        print(X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i+1]
    imfsValues_pre.append(predictions)
  
  imfsValues_pre = pd.DataFrame(imfsValues_pre)
  imfsValues_pre_test = imfsValues_pre.transpose()
  imfsValues_pre_test['residue'] = RESIDUE[-30:]
  imfsValues_pre_test['sum'] = imfsValues_pre_test.apply(lambda x:x.sum(),axis = 1)
  pre = np.array(imfsValues_pre_test['sum'])
  mape = mean_absolute_percentage_error(df_epalia_c_d_1_value[-30:], pre)  
  return imfsValues_pre_test, mape

# rmse = []
# for param in combinations:
#   pre,rmse_temp = lstm_pre_eemd(imfsValues,RESIDUE,param['epochs'],param['neurons'],param['node'])
#   rmse.append(rmse_temp)
#   print(rmse)
  
# combinations[np.argmin(rmse)]

# param = combinations[np.argmin(rmse)]

# imfsValues_pre_final, rmse_final = lstm_pre_eemd(imfsValues,RESIDUE,param['epochs'],param['neurons'],param['node'])

imfsValues_pre_final, rmse_final = lstm_pre_eemd_2(imfsValues,RESIDUE,17,10,15)
pre = np.array(imfsValues_pre_final['sum'])
mape = mean_absolute_percentage_error(df_epalia_c_d_1_value[-30:], pre)
mape_percent = 100*mape
print('Test MAPE:%.3f' % mape_percent)
true_value = df_epalia_c_d_1_value[-30:]

pyplot.plot(true_value)
pyplot.plot(pre)
pyplot.show()

plt.figure(figsize=(24,8),dpi=80)
date = price_selected.loc[1000:,['date']]
date = list(date['date'])
date_pre = price_selected.loc[price_selected.tail(30).index,['date']]
date_pre = list(date_pre['date'])
collect = price_selected.loc[1000:,['close']]
# collect_pre = df_epalia_c_d_1.loc[train_index_group[i],['Nb de palettes totales']]
# collect_pre = list(collect_pre['Nb de palettes totales'])
collect_pre = pre
collect_pre = list(collect_pre)

# pre_temp = collect_pre.extend(pre_total[i])
plt.plot_date(date,collect,fmt='b.',linestyle='solid',label = 'real_close_price' )
plt.plot_date(date_pre,collect_pre,fmt='b.',color='r',linestyle='solid',label = 'pre_close_price')
plt.legend(labels=['real_close_price','pre_close_price'],loc='best',fontsize = 15)

plt.title('Real Close Prices and Pre Close Prices',fontsize=15)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(8))
plt.xticks(rotation = 90,ha = 'center')


