import yfinance as yf
import numpy as np
import pandas as pd

from numpy import concatenate, array
from pandas import concat, DataFrame

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras import optimizers
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, save_model, load_model
from keras.callbacks import EarlyStopping
import keras_tuner
from keras_tuner import RandomSearch

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics

import math
from math import sqrt
import time
import math
import json
import os
from os import path
import gc
import shutil

from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings(action="ignore")
import time

# downloading data
def get_data(index_symbol, start = "2000-01-01", end="2023-08-31"):
    df_index = yf.download(index_symbol, start, end).dropna()
    return df_index

# seperation of lag data set 
def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def fig_LSTM(df_index, df_test, df_predictions):   
    fig_train_test_vs_prediction = go.Figure()

    fig_train_test_vs_prediction.add_trace(
        go.Scatter(x=df_index.index, y=df_index['Close'], name="The whole dataset"),
    )

    fig_train_test_vs_prediction.add_trace(
        go.Scatter(x=df_test['Date'], y=df_test['inv_y__test'], name="Test"),
    )

    fig_train_test_vs_prediction.add_trace(
        go.Scatter(x=df_predictions['Date'], y=df_predictions['inv_y_pred_test'], name="Prediction")
    )

    fig_train_test_vs_prediction.update_layout(
        title={
            'text': "Display of real values vs Predictions",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Closing Price",
        legend_title="Data",
        template="plotly_dark"

    )

    fig_train_test_vs_prediction.show()

def transaction_cost(df_equity_curve, transaction_cost):

    prev_position = df_equity_curve['position'].iloc[0]

    for i, position in enumerate(df_equity_curve['position']):
        if position != prev_position:
            df_equity_curve['strat_return'].iloc[i] -= transaction_cost        
        prev_position = position
    
    return df_equity_curve

def fig_strategies(df_equity_curve):
    fig_equity_curve_strategy = go.Figure()

    fig_equity_curve_strategy.add_trace(
        go.Scatter(x=df_equity_curve.index, y=df_equity_curve['strategy'], name="equity_curve_strategy"),
    )

    fig_equity_curve_strategy.add_trace(
        go.Scatter(x=df_equity_curve.index, y=df_equity_curve['buy_n_hold'], name="Benchmark"),
    )

    fig_equity_curve_strategy.update_layout(
        title={
            'text': "Strategy",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Data",
        template="plotly_dark"

    )

    return fig_equity_curve_strategy.show()

# performance metrics

def EquityCurve_na_StopyZwrotu(tab):

    ret=[]

    for i in range(0,len(tab)-1):
        ret.append((tab[i+1]/tab[i])-1)

    return ret

def ARC(tab):
    temp=EquityCurve_na_StopyZwrotu(tab)
    lenth = len(tab)
    a_rtn=1
    for i in range(len(temp)-1):
        rtn=(1+temp[i])
        a_rtn=a_rtn*rtn
    if a_rtn <= 0:
        a_rtn = 0
    else:
        a_rtn = math.pow(a_rtn,(252/lenth)) - 1
    return 100*a_rtn


def MaximumDrawdown(tab):

    eqr = np.array(EquityCurve_na_StopyZwrotu(tab))
    
    cum_returns = np.cumprod(1 + eqr)
    cum_max_returns = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_max_returns - cum_returns) / cum_max_returns
    max_drawdown = np.max(drawdowns)
    return max_drawdown*100


def ASD(tab):
    return ((((252)**(1/2)))*np.std(EquityCurve_na_StopyZwrotu(tab)))*100

def sgn(x):
    if x==0:
        return 0
    else:
        return int(abs(x)/x) #int(abs(x)/x)

def MLD(temp):
    if temp==[]:
        return 1
    if temp !=[]:
        i=np.argmax(np.maximum.accumulate(temp)-temp)
        if i==0:
            return len(temp)/252.03
        j=np.argmax(temp[:i]) 
        MLD_end=-1
        for k in range(i,len(temp)):
            if (temp[k-1] < temp[j]) and (temp[j]<temp[k]):
                MLD_end=k
                break
        if MLD_end == -1:
            MLD_end=len(temp)

    return abs(MLD_end - j)/252.03

def IR1(tab):

    aSD=ASD(tab)

    ret=ARC(tab)#/100

    licznik=ret

    mianownik=aSD

    val=licznik/mianownik

    if mianownik==0:
        return 0
    else: 
        return max(val,0)

def IR2(tab):

    aSD=ASD(tab)

    ret=ARC(tab)#/100

    md=MaximumDrawdown(tab)#/100

    licznik=(ret**2)*sgn(ret)

    mianownik=aSD*md

    val=licznik/mianownik

    if mianownik==0:
        return 0
    else: 
        return max(val,0)


def wyniki(tab, nazwa):
    print('Wyniki dla {0} prezentują się następująco: \n'.format(nazwa))
    print('ASD {0} \n MD {1}% \n ARC {2}% \n MLD {3} lat \n IR1 {4} \n IR2 {5}'.format("%.2f" % ASD(tab),"%.2f" %  MaximumDrawdown(tab),"%.2f" %  ARC(tab),"%.2f" %  MLD(tab),"%.4f" % IR1(tab),"%.4f" % IR2(tab))) 

def porownanie(tab_Algo, tab_BH):

    ASD_bh = ASD(tab_BH)
    ASD_alg = ASD(tab_Algo)

    MD_bh = MaximumDrawdown(tab_BH)
    MD_alg = MaximumDrawdown(tab_Algo)

    ARC_bh = ARC(tab_BH)
    ARC_alg = ARC(tab_Algo)

    MLD_bh = MLD(tab_BH)
    MLD_alg = MLD(tab_Algo)

    IR1_bh = IR1(tab_BH)
    IR1_alg = IR1(tab_Algo)

    IR2_bh = IR2(tab_BH)
    IR2_alg = IR2(tab_Algo)

    if ASD_bh>=ASD_alg:
        print('Strategia lepsza od BH pod względem ASD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem ASD')

    if ARC_alg>=ARC_bh:
        print('\033[92m Strategia lepsza od BH pod względem ARC')
    else:
        print('\033[91m Strategia gorsza od BH pod względem ARC')

    if MD_bh>=MD_alg:
        print('\033[92m Strategia lepsza od BH pod względem MD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem MD')

    if MLD_bh>=MLD_alg:
        print('\033[92m Strategia lepsza od BH pod względem MLD')
    else:
        print('\033[91m Strategia gorsza od BH pod względem MLD')

    if IR1_alg>=IR1_bh:
        print('\033[92m Strategia lepsza od BH pod względem IR1')
    else:
        print('\033[91m Strategia gorsza od BH pod względem IR1')

    if IR2_alg>=IR2_bh:
        print('\033[92m Strategia lepsza od BH pod względem IR2')
    else:
        print('\033[91m Strategia gorsza od BH pod względem IR2')

class HyperparameterTuner:
    def __init__(self, DATASETS, lookback_bars, validation_bars, testing_bars, stock_name, EPOCHS, PATIENCE, TRIALS):
        self.DATASETS = DATASETS
        self.lookback_bars = lookback_bars
        self.validation_bars = validation_bars
        self.testing_bars = testing_bars
        self.stock_name = stock_name
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.TRIALS = TRIALS

        self.hp = keras_tuner.HyperParameters()
        self.sequence_length = self.hp.Choice('sequence_length', values=[7, 14, 21])

    def build_model_LSTM(self, hp):
        model = Sequential()
        DROPOUT = 0.075
        LSTM_ACTIVATION = 'sigmoid' # change to 'tanh'
        ACTIVATION = 'tanh' # change to 'linear'
        LOSS = 'mean_squared_error'
        
        sequence_lengths = [7, 14, 21]
        sequence_length = hp.Choice('sequence_length', values=sequence_lengths)

        hp_neurons = hp.Choice('neurons', values=[25, 50, 75, 100, 250, 500])
        hp_layers = hp.Choice('layers', values=[0, 1])
        hp_optimizer = hp.Choice('optimizer', values=['Adam', 'Nadam', 'Adagrad'])
        hp_learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.01])

        model.add(LSTM(units=hp_neurons, input_shape=(self.sequence_length, 4), return_sequences=True))
        model.add(Dropout(DROPOUT))

        for _ in range(hp_layers):
            model.add(LSTM(units=hp_neurons, return_sequences=True))
            model.add(Dropout(DROPOUT))

        model.add(LSTM(units=hp_neurons, return_sequences=False))
        model.add(Dropout(DROPOUT))

        model.add(Dense(1, activation=ACTIVATION))

        if hp_optimizer == 'Adam':
            OPT = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'Nadam':
            OPT = tf.keras.optimizers.Nadam(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'Adagrad':
            OPT = tf.keras.optimizers.Adagrad(learning_rate=hp_learning_rate)

        model.compile(loss=LOSS, optimizer=OPT, metrics=['accuracy'])

        return model

    def hyperparameter_tuning(self):
        ranges = list(range(self.lookback_bars, len(self.DATASETS[21][0]) - self.testing_bars, self.validation_bars))
        for i in range(0, len(ranges)):

            orignal_Sequence_Length = self.sequence_length

            tuner = RandomSearch(
                self.build_model_LSTM,
                objective='val_loss',
                max_trials=self.TRIALS,
                executions_per_trial=1,
                directory=f'./Hyperparameter_tunning/{self.stock_name}/RS_{i}'
            )
            data_X, data_Y = self.DATASETS[self.sequence_length][0], self.DATASETS[self.sequence_length][1]

            train_X, train_y = data_X[ranges[i]-self.lookback_bars:ranges[i]], data_Y[ranges[i]-self.lookback_bars:ranges[i]]
            val_X, val_Y = data_X[ranges[i]:ranges[i]+self.validation_bars], data_Y[ranges[i]:ranges[i]+self.validation_bars]
            test_X, test_y = data_X[ranges[i]+self.validation_bars:ranges[i]+self.validation_bars+self.testing_bars], data_Y[ranges[i]+self.validation_bars:ranges[i]+self.validation_bars+self.testing_bars]
            
            early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.PATIENCE)

            tuner.search(train_X, train_y, epochs=self.EPOCHS, validation_data=(val_X, val_Y), callbacks=[early_stop], batch_size=64)

            for num in range(5):
                best_hps = tuner.get_best_hyperparameters(num_trials=self.TRIALS)[num]
                best_sequence_length = best_hps.get('sequence_length')
            
                data_X_BEST, data_Y_BEST = self.DATASETS[best_sequence_length][0], self.DATASETS[best_sequence_length][1]

                train_X_BEST, train_y_BEST = data_X_BEST[ranges[i]-self.lookback_bars:ranges[i]], data_Y_BEST[ranges[i]-self.lookback_bars:ranges[i]]
                val_X_BEST, val_Y_BEST = data_X_BEST[ranges[i]:ranges[i]+self.validation_bars], data_Y_BEST[ranges[i]:ranges[i]+self.validation_bars]
                test_X_BEST, test_y_BEST = data_X_BEST[ranges[i]+self.validation_bars:ranges[i]+self.validation_bars+self.testing_bars], data_Y_BEST[ranges[i]+self.validation_bars:ranges[i]+self.validation_bars+self.testing_bars]

                self.sequence_length = best_sequence_length
                model_best = tuner.hypermodel.build(best_hps)

                if path.exists(f'./Hyperparameter_tunning/{self.stock_name}/HP_Grid_Search_{i}/model_GS_{num}.h5') == True:
                    print(f'model_{num} EXISTS for id_{i} path exits') 
                else:
                    model_best.fit(train_X_BEST, train_y_BEST, epochs=self.EPOCHS, validation_data=(val_X_BEST, val_Y_BEST), callbacks=[early_stop], batch_size=64)
                    model_best.save(f'./Hyperparameter_tunning/{self.stock_name}/HP_Grid_Search_{i}/model_GS_{num}.h5')

                print(f"model_{num}: {best_sequence_length}")
                
            self.sequence_length = orignal_Sequence_Length

            tf.keras.backend.clear_session()

            print(f"id_{i}")

        return tuner

def predict_LSTM(DATASETS, lookback_bars, validation_bars, testing_bars, stock_name):

    test_model_predictions = []
    TEST_Y = []

    ranges = list(range(lookback_bars, len(DATASETS[21][0]) - testing_bars, validation_bars))
    for i in range(0, len(ranges)):

        model = load_model(f"./Best_Models/{stock_name}/model_ID_{i}.h5")
        input_timesteps = model.get_config()['layers'][0]['config']['batch_input_shape'][1]
        
        DATA_X, DATA_Y = DATASETS[input_timesteps][0], DATASETS[input_timesteps][1]
        train_X, train_y = DATA_X[ranges[i]-lookback_bars:ranges[i]], DATA_Y[ranges[i]-lookback_bars:ranges[i]], 
        val_X, val_Y = DATA_X[ranges[i]:ranges[i]+validation_bars], DATA_Y[ranges[i]:ranges[i]+validation_bars], 
        test_X, test_y = DATA_X[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars], DATA_Y[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars]

        test_model_predictions.append(model.predict(test_X))
        TEST_Y.append(test_y)   
        
        print(test_X.shape, test_y.shape, f"id_{i}")

    return TEST_Y, test_model_predictions

def get_data_ARIMA(index_symbol, start, end="2023-08-31"):
    df_index = yf.download(index_symbol, start, end).dropna()
    df_index = df_index[['Close']]
    return df_index

# creating arima_config
def arima_configuration(p_max, q_max):
    orders = []
    for p in range(0,p_max):
        for q in range(0,q_max):
            if p!=q:
                orders.append((p, 1, q))
    return orders
    
# finding ARIMA order AIC and creating a dict
def get_order_aic(arima_orders, data):
    order_choose = dict()
    for ord in arima_orders:
        model = ARIMA(data['Close'], order=ord)
        model.initialize_approximate_diffuse() 
        model_fit = model.fit()
        order_choose[ord] = model_fit.aic
    return order_choose

# arima prediction creation
def arima_predict(history, order_choose):
    model = ARIMA(history, order=min(order_choose, key=order_choose.get))
    model.initialize_approximate_diffuse() 
    model_fit = model.fit()
    output = model_fit.forecast()
    y_hat = output[0]
    return y_hat

# combination of fucntions and the main results
def main_arima(p_max, q_max, df_index, lookback_bars, validation_bars, testing_bars): 

    model_predictions = []
    train_data = []
    test_data_ = []
    ranges = list(range(lookback_bars + 0, len(df_index) - 0, validation_bars))

    arima_orders = arima_configuration(p_max, q_max)

    for i in range(0, len(ranges)): #len(ranges)
        training_data = df_index.iloc[ranges[i]-lookback_bars:ranges[i]]
        test_data = df_index.iloc[ranges[i]:ranges[i]+validation_bars]
        # test_data = df_index.iloc[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars]

        data = pd.concat([training_data], axis=0)

        order_aic = get_order_aic(arima_orders, data)

        history = [tr for tr in data['Close']] #.dropna()]
        N_test_observations = len(test_data)
        for time_point in range(N_test_observations):
            y_hat = arima_predict(history, order_aic)
            model_predictions.append(y_hat)
            real_test_value = test_data['Close'][time_point]
            history.append(real_test_value)
        
        train_data.append(training_data)
        test_data_.append(test_data)

        print(len(training_data), len(test_data))

    return model_predictions, train_data, test_data_

########################################################################

def IR_calc_train_val(scaler_Y, model, VALUE_X, VALUE_Y, TRANSACTON_COST, POS):
      # IR2 TRAIN

      value_predic = scaler_Y.inverse_transform(model.predict(VALUE_X)).flatten()
      value_real = scaler_Y.inverse_transform(VALUE_Y).flatten()

      df = pd.DataFrame(
                        data={'value_predic': value_predic,
                              'value_real': value_real}
                        )

      position= np.where(df['value_predic'].shift(-1)>df['value_real'],1,POS)
      df['position'] = position
      df['strat_return'] = df['value_real'].pct_change().dropna()
      
      df = transaction_cost(df, TRANSACTON_COST)

      df["strategy"] = (df["strat_return"] * df['position'].shift(1))
      df["strategy"] = (1 + df["strategy"].fillna(0)).cumprod()
      
      IR_2_value = IR2(df["strategy"].values)

      return IR_2_value

def find_best_model_LO(DATASETS, lookback_bars, validation_bars, testing_bars, stock_name, scaler_Y, TRANSACTON_COST, TRIALS):
    # ir2_best_model_train = {}
    
    ranges = list(range(lookback_bars, len(DATASETS[21][0]) - testing_bars, validation_bars))

    df = pd.DataFrame()
    
    id_ = []
    model_num = []
    IR_2_train_value = []
    IR_2_validation_value = []
    IR_2_test_value = []

    loss_train_value = []
    loss_validation_value = []
    
    for i in range(0, len(ranges)): 
        
        if path.exists(f'./Best_Models/{stock_name}/Long-Only/model_ID_{i}.h5') == True:
            print(f'[SECTION_1 --> id_{i}] path exits') 
        
        else:
            for num in range(5):
                model = load_model(f'./Hyperparameter_tunning/{stock_name}/HP_Grid_Search_{i}/model_GS_{num}.h5')
                input_timesteps = model.get_config()['layers'][0]['config']['batch_input_shape'][1]

                DATA_X, DATA_Y = DATASETS[input_timesteps][0], DATASETS[input_timesteps][1]
                train_X, train_y = DATA_X[ranges[i]-lookback_bars:ranges[i]], DATA_Y[ranges[i]-lookback_bars:ranges[i]], 
                val_X, val_Y = DATA_X[ranges[i]:ranges[i]+validation_bars], DATA_Y[ranges[i]:ranges[i]+validation_bars], 
                test_X, test_y = DATA_X[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars], DATA_Y[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars]

                IR_2_train = IR_calc_train_val(scaler_Y, model, train_X, train_y, TRANSACTON_COST, 0)
                IR_2_validation  = IR_calc_train_val(scaler_Y, model, val_X, val_Y, TRANSACTON_COST, 0)
                IR_2_test  = IR_calc_train_val(scaler_Y, model, test_X, test_y, TRANSACTON_COST, 0)

                loss_train = model.evaluate(train_X, train_y)[0]
                loss_validation = model.evaluate(val_X, val_Y)[0]

                id_.append(f'id_{i}')

                model_num.append(f'model_{num}')

                IR_2_train_value.append(IR_2_train)
                IR_2_validation_value.append(IR_2_validation)
                IR_2_test_value.append(IR_2_test)

                loss_train_value.append(loss_train)
                loss_validation_value.append(loss_validation)

                print(f'id_{i}, model_{num}, IR_2_train = {IR_2_train}, IR_2_validation = {IR_2_validation}, IR_2_test = {IR_2_test}, loss_train = {loss_train}, loss_validation = {loss_validation}')

        print(f"id_{i}")

    df = pd.DataFrame(data={
        'id_': id_,
        'model_num': model_num,
        'IR_2_train_value': IR_2_train_value,
        'IR_2_validation_value': IR_2_validation_value,
        'IR_2_test_value': IR_2_test_value,
        'loss_train_value': loss_train_value,
        'loss_validation_value': loss_validation_value,
    })
        
    return df

def find_best_model_LS(DATASETS, lookback_bars, validation_bars, testing_bars, stock_name, scaler_Y, TRANSACTON_COST, TRIALS):
    
    ranges = list(range(lookback_bars, len(DATASETS[21][0]) - testing_bars, validation_bars))

    df = pd.DataFrame()
    
    id_ = []
    model_num = []
    IR_2_train_value = []
    IR_2_validation_value = []
    IR_2_test_value = []

    loss_train_value = []
    loss_validation_value = []
    
    for i in range(0, len(ranges)): 
        
        if path.exists(f'./Best_Models/{stock_name}/Long-Short/model_ID_{i}.h5') == True:
            print(f'[SECTION_1 --> id_{i}] path exits') 
        
        else:
            for num in range(5):
                model = load_model(f'./Hyperparameter_tunning/{stock_name}/HP_Grid_Search_{i}/model_GS_{num}.h5')
                input_timesteps = model.get_config()['layers'][0]['config']['batch_input_shape'][1]

                DATA_X, DATA_Y = DATASETS[input_timesteps][0], DATASETS[input_timesteps][1]
                train_X, train_y = DATA_X[ranges[i]-lookback_bars:ranges[i]], DATA_Y[ranges[i]-lookback_bars:ranges[i]], 
                val_X, val_Y = DATA_X[ranges[i]:ranges[i]+validation_bars], DATA_Y[ranges[i]:ranges[i]+validation_bars], 
                test_X, test_y = DATA_X[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars], DATA_Y[ranges[i]+validation_bars:ranges[i]+validation_bars+testing_bars]

                IR_2_train = IR_calc_train_val(scaler_Y, model, train_X, train_y, TRANSACTON_COST, -1)
                IR_2_validation = IR_calc_train_val(scaler_Y, model, val_X, val_Y, TRANSACTON_COST, -1)
                IR_2_test = IR_calc_train_val(scaler_Y, model, test_X, test_y, TRANSACTON_COST, -1)

                loss_train = model.evaluate(train_X, train_y)[0]
                loss_validation = model.evaluate(val_X, val_Y)[0]

                id_.append(f'id_{i}')

                model_num.append(f'model_{num}')

                IR_2_train_value.append(IR_2_train)
                IR_2_validation_value.append(IR_2_validation)
                IR_2_test_value.append(IR_2_test)

                loss_train_value.append(loss_train)
                loss_validation_value.append(loss_validation)

                print(f'id_{i}, model_{num}, IR_2_train = {IR_2_train}, IR_2_validation = {IR_2_validation}, IR_2_test = {IR_2_test}, loss_train = {loss_train}, loss_validation = {loss_validation}')

        print(f"id_{i}")

    df = pd.DataFrame(data={
        'id_': id_,
        'model_num': model_num,
        'IR_2_train_value': IR_2_train_value,
        'IR_2_validation_value': IR_2_validation_value,
        'IR_2_test_value': IR_2_test_value,
        'loss_train_value': loss_train_value,
        'loss_validation_value': loss_validation_value,
    })
        
    return df