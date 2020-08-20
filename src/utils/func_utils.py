import pandas as pd
import numpy as np
import datetime
import os

import src.utils.indicators as indicators
import fix_yahoo_finance as yf

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def getData(data_name):

    print("Cargando datos...")

    path_data = '../data/'+data_name+'.csv'
    df = None

    # Check if data exists
    # If not exists then data is downloaded and save in folder data
    if os.path.exists(path_data):
        print('Datos existentes en ../data.')
        df = pd.read_csv(path_data, index_col = "Date", parse_dates = True)
        print(path_data + ' cargado con éxito.')
    else:
        print('Datos no existentes en ../data.')
        print('Descargando datos..')
        from_date = '2000-01-01'
        today = datetime.datetime.now()
        today = today.strftime('%Y-%m-%d')

        df = yf.download(data_name, from_date, today)
        df = df[['Open','High', 'Low', 'Close', 'Volume']]

        if not os.path.exists('../data'):
            os.makedirs('../data')

        df.to_csv(path_data)

        print('Datos ' + path_data + ' guardados.')

    return df


def add_features(df):
    """
    Add to df dataframe new features with technical indicators
    :param df: dataframe with market data
    :return: dataframe with the new features added
    """

    print("Añadiendo características...")
    # Momento
    df = indicators.momentum(df, 5)
    df = indicators.momentum(df, 10)
    df = indicators.momentum(df, 15)

    # Media Movil
    df = indicators.moving_average(df, 7)
    df = indicators.moving_average(df, 14)
    df = indicators.moving_average(df, 21)

    # Media Exponencial
    df = indicators.exponential_moving_average(df, 7)
    df = indicators.exponential_moving_average(df, 14)
    df = indicators.exponential_moving_average(df, 21)

    # Rate of change
    df = indicators.rate_of_change(df, 13)
    df = indicators.rate_of_change(df, 21)

    # Oscilador estocastico
    df = indicators.stochastic(df, 7)
    df = indicators.stochastic(df, 14)
    df = indicators.stochastic(df, 21)

    # Oscilador estocastico fast
    df = indicators.stochastic_fast(df, 7)
    df = indicators.stochastic_fast(df, 14)
    df = indicators.stochastic_fast(df, 21)

    # MACD e histograma
    df = indicators.moving_average_CD(df, 12, 26)

    # Indice de fuerza relativa
    df = indicators.relative_strength_index(df, 9)
    df = indicators.relative_strength_index(df, 14)
    df = indicators.relative_strength_index(df, 21)

    # Desviacion tipica
    df = indicators.standard_deviation(df, 7)
    df = indicators.standard_deviation(df, 14)
    df = indicators.standard_deviation(df, 21)

    df = df.dropna()

    return df


def add_label(df, gain, loss ,n_day, commission):
    """
    Add a label to each day of the dataframe
    0 - Sell, 1 - Buy

    :param df: dataframe with market data
    :param gain: gain limit
    :param loss: loss limit
    :param n_day: number of days of the simulation
    :param commission: commission considerated for the simulation
    :return: dataframe with labels added
    """

    print('Añadiendo etiquetas...')

    df_closes = [df.iloc[i]['Close'] for i in range(len(df.index))]
    labels = []

    # Iterate over the historical data
    for i in range(len(df.index)-n_day):
        close_price = df_closes[i]
        # Iterate the next n_day days
        for j in range(i+1, i+1+n_day):
            dif = (df_closes[j] - close_price)/close_price
            # Exit if limit is crossed
            if dif > gain or dif < -loss:
                break

        action = 1 if dif > commission*2 else 0
        labels.append(action)

    # The last n_day days in the historical data are labeled as 0 (Sell)
    labels.extend([0 for x in range(n_day)])
    df['label'] = labels

    return df

def encode_to_categorical(y):
    """
    Convert to categorical
    :param y: vector to encode to categorical
    :return: vector of categorical features
    """

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    y = np_utils.to_categorical(encoded_y)

    return y

def split_df_date(df, start_train_date, end_train_date, start_test_date, end_test_date):
    """
    Split dataframe in train and test from given dates
    :param df: dataframe with market data
    :param start_train_date: start date of the training dataset
    :param end_train_date: end date of the training dataset
    :return: train and test datasets
    """

    start_date = datetime.datetime.strptime(str(df.index.date[0]), '%Y-%m-%d')
    end_date = datetime.datetime.strptime(str(df.index.date[len(df.index)-1]), '%Y-%m-%d')

    df = df[start_train_date:end_test_date]
    df_train, df_test = df[start_train_date:end_train_date], df[start_test_date:end_test_date]

    train_size = df_train.shape[0]
    test_size = df_test.shape[0]

    # Definimos la matriz de caracteristicas excluyendo la etiqueta y el vector de etiquetas
    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])

    # Separamos en train y test
    X_train, X_test = X[0:train_size], X[len(X)-test_size:len(X)]
    y_train, y_test = y[0:train_size], y[len(X)-test_size:len(X)]

    return df_train, df_test, X_train, X_test, y_train, y_test


def get_split_w_threshold(alpha, normalization='exponential'):
    """
    Get normalize weights and thresholds from alpha vector
    :param alpha: optimize Vectorize
    :return: weights and thresholds
    """
    w = []

    if normalization == 'exponential':
        w = np.exp(alpha[:len(alpha)-2])/np.sum(np.exp(alpha[:len(alpha)-2]))
    elif normalization == 'l1':
        w = alpha[:len(alpha)-2]/np.sum(np.abs(alpha[:len(alpha)-2]))

    buy_threshold = alpha[len(alpha)-2]
    sell_threshold = alpha[len(alpha)-1]

    return w, buy_threshold, sell_threshold


def get_combined_signal(moving_average_rules, moving_averages, w, index):
    """
    Combines in a weighted way buy-sell signals coming from moving average crosses.
    :param moving_average_rules: list with moving average rules
    :param moving_averages: dict with moving averages from historical data
    :param w: weights vector
    :parm index: moving averages index
    :return: final signal get from combined all signals
    """
    signal_list = []

    # Get signals from all moving averages rules
    for short_period, long_period in moving_average_rules:
        moving_average_short = moving_averages['MA_' + str(short_period)][index]
        moving_average_long = moving_averages['MA_' + str(long_period)][index]

        if moving_average_short < moving_average_long:
            signal_list.append(-1)
        else:
            signal_list.append(+1)

    final_signal = np.sum(np.array(w)*np.array(signal_list))

    return final_signal
