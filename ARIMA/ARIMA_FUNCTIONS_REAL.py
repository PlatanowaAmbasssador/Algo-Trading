import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff

from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings(action="ignore")
import time
import math

import os
import pickle
from typing import Tuple

#  FUNCTIONS

# fig of real data set vs. the predictions

def get_data_ARIMA(index_symbol, start, end = "2023-08-31"):
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

def fig_ARIMA(df_index, df_test, df_predictions):   
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

