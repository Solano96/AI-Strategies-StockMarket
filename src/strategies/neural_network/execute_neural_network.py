# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Import utils function
import src.utils.func_utils as func_utils

# Import classes
from src.classes.myCerebro import MyCerebro
from src.classes.myAnalyzer import MyAnalyzer
from src.classes.myBuySell import MyBuySell
from src.classes.maxRiskSizer import MaxRiskSizer

# Import strategies execution
import src.strategies_execution.execution_analysis as execution_analysis
import src.strategies_execution.execution_plot as execution_plot
from src.strategies_execution.executions import print_execution_name
from src.strategies_execution.executions import execute_strategy

# Import strategy
from src.strategies.neural_network.class_neural_network import NeuralNetwork
from src.strategies.neural_network.strategy_neural_network import NeuralNetworkStrategy


def execute_neural_network_strategy(df, options, commission, data_name, start_date, end_date):
    """
    Execute neural network strategy on data history contained in df
    :param df: dataframe with historical data
    :param options: dict with the following parameters
        - gain - gain threshold in simulation of labelling
        - loss - loss threshold simulation of labelling
        - n_day - number of days in simulation of labelling
        - epochs - number of epochs to train the neural network
    :param commission: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - NN_Cerebro - execution engine
        - NN_Strategy - neural network strategy instance
    """

    print_execution_name("Estrategia: red neuronal")

    strategy_name = 'red_neuronal'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    # Get parameters
    gain = options['gain']
    loss = options['loss']
    n_day = options['n_day']
    epochs = options['epochs']

    s_test_date = datetime.strptime(start_date, '%Y-%m-%d')
    s_train = s_test_date.replace(year = s_test_date.year - 2)
    e_train = s_test_date - timedelta(days=1)

    # Preprocess dataset
    df = func_utils.add_features(df)
    df = func_utils.add_label(df, gain = gain, loss = loss, n_day = n_day, commission = commission)

    # Split train and test
    df_train, df_test, X_train, X_test, y_train, y_test = func_utils.split_df_date(df, s_train, e_train, start_date, end_date)

    # Normalization
    print("Normalizando datos...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform (X_test)

    # Transform data in a correct format to use in Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get prediction model
    print("Entrenando red neuronal...")
    neural_network = NeuralNetwork()
    neural_network.build_model(input_shape = (X_train.shape[1], 1))
    neural_network.train(X_train, y_train, epochs = epochs)

    # Get accuraccy
    train_accuracy = neural_network.get_accuracy(X_train, y_train)
    test_accuracy = neural_network.get_accuracy(X_test, y_test)

    print("\nRESULTADOS PREDICCION:\n")
    print("TRAIN :: Porcentaje de acierto: " + str(train_accuracy))
    print("TEST  :: Porcentaje de acierto: " + str(test_accuracy))

    # ------------------------ Backtesting ------------------------ #

    # Initialize neural network memory
    neural_network.init_memory(X_train[len(X_train)-15:len(X_train)], y_train[len(y_train)-15:len(y_train)])

    # Create an instance from NeuralNetworkStrategy class and assign parameters
    NN_Strategy = NeuralNetworkStrategy
    NN_Strategy.X_test = X_test
    NN_Strategy.y_test = y_test
    NN_Strategy.model = neural_network
    NN_Strategy.n_day = n_day

    # Execute strategy
    NN_Cerebro = execute_strategy(NN_Strategy, df_test, commission, info, options)

    return NN_Cerebro, NN_Strategy
