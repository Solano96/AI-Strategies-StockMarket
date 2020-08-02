# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys, getopt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler

import utils.func_utils as func_utils
import utils.myCerebro as myCerebro
import utils.myAnalyzer as myAnalyzer
import utils.testStrategy as testStrategy
import utils.model as model
import utils.strategies as strategies
import utils.geneticRepresentation as geneticRepresentation

import strategies_execution.execution_analysis as execution_analysis
import strategies_execution.execution_plot as execution_plot

import pyswarms as ps

import backtrader as bt
import backtrader.plot
import matplotlib
import matplotlib.pyplot as plt

from numpy.random import seed

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

seed(1)
# Opciones de ejecucion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=sys.maxsize)


def print_execution_name(execution_name):
    print("\n --------------- ", execution_name, " --------------- \n")


def execute_buy_and_hold_strategy(df, commission, data_name, start_date, end_date):

    print_execution_name("Estrategia: comprar y mantener")

    df = df[start_date:end_date]

    BH_Strategy =  strategies.BuyAndHoldStrategy
    BH_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(BH_Strategy, df, commission)

    # Save results
    execution_analysis.printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'comprar_y_mantener')
    # Save simulation chart
    execution_plot.plot_simulation(BH_Cerebro, 'comprar_y_mantener', data_name, start_date, end_date)

    return BH_Cerebro, BH_Strategy


def execute_classic_strategy(df, commission, data_name, start_date, end_date):

    print_execution_name("Estrategia: clásica")

    df = df[start_date:end_date]

    Classic_Strategy =  strategies.ClassicStrategy
    Classic_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(Classic_Strategy, df, commission)

    # Save results
    execution_analysis.printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'estrategia_clasica')
    # Save simulation chart
    execution_plot.plot_simulation(Classic_Cerebro, 'estrategia_clasica', data_name, start_date, end_date)

    return Classic_Cerebro, Classic_Strategy


def execute_neural_network_strategy(df, options, comm, data_name, s_test, e_test):

    print_execution_name("Estrategia: red neuronal")

    # ------------ Get parameters ------------#
    gain = options['gain']
    loss = options['loss']
    n_day = options['n_day']
    epochs = options['epochs']

    s_test_date = datetime.strptime(s_test, '%Y-%m-%d')
    s_train = s_test_date.replace(year = s_test_date.year - 2)
    e_train = s_test_date - timedelta(days=1)

    # ------------ Preprocess dataset ------------ #

    df = func_utils.add_features(df)
    df = func_utils.add_label(df, gain = gain, loss = loss, n_day = n_day, commission = comm)

    # ------------ Split train and test ------------ #

    df_train, df_test, X_train, X_test, y_train, y_test = func_utils.split_df_date(df, s_train, e_train, s_test, e_test)

    # ------------ Normalization ------------ #

    print("Normalizando datos...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform (X_test)

    # Transform data in a correct format to use in Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # ------------ Get prediction model ------------ #

    print("Entrenando red neuronal...")

    neural_network = model.NeuralNetwork()
    neural_network.build_model(input_shape = (X_train.shape[1], 1))
    neural_network.train(X_train, y_train, epochs = epochs)

    # ------------ Get accuraccy ------------ #

    train_accuracy = neural_network.get_accuracy(X_train, y_train)
    test_accuracy = neural_network.get_accuracy(X_test, y_test)

    print("\nRESULTADOS PREDICCION:\n")
    print("TRAIN :: Porcentaje de acierto: " + str(train_accuracy))
    print("TEST  :: Porcentaje de acierto: " + str(test_accuracy))

    # ------------------------ Backtesting ------------------------ #

    # Initialize neural network memory
    neural_network.init_memory(X_train[len(X_train)-15:len(X_train)], y_train[len(y_train)-15:len(y_train)])

    # Create an instance from NeuralNetworkStrategy class and assign parameters
    NN_Strategy = strategies.NeuralNetworkStrategy
    NN_Strategy.X_test = X_test
    NN_Strategy.y_test = y_test
    NN_Strategy.model = neural_network
    NN_Strategy.n_day = n_day

    # Execute strategy
    NN_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(NN_Strategy, df_test, comm)
    # Save results
    execution_analysis.printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'red_neuronal', train_accuracy, test_accuracy)
    # Save simulation chart
    execution_plot.plot_simulation(NN_Cerebro, 'red_neuronal', data_name, s_test, e_test)

    return NN_Cerebro, NN_Strategy


def execute_pso_strategy(df, commission, data_name, s_test, e_test):

    print_execution_name("Estrategia: particle swar optimization")

    # ------------ Obtenemos los conjuntos de train y test ------------ #

    s_test_date = datetime.strptime(s_test, '%Y-%m-%d')
    s_train = s_test_date.replace(year = s_test_date.year - 2)
    e_train = s_test_date - timedelta(days=1)

    gen_representation = geneticRepresentation.GeneticRepresentation(df, s_train, e_train, s_test, e_test)

    # ------------ Fijamos hiperparámetros ------------ #

    n_particles=20
    dimensions=107
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    max_bound = 1.0 * np.ones(dimensions-2)
    min_bound = -max_bound
    max_bound = np.append(max_bound, [1.0, 0.0])
    min_bound = np.append(min_bound, [0.0, -1.0])
    bounds = (min_bound, max_bound)
    iters = 100

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)

    # Perform optimization
    kwargs={'from_date': s_train, 'to_date': e_train}
    best_cost, best_pos = optimizer.optimize(gen_representation.cost_function, iters=iters, **kwargs)

    # Create an instance from CombinedSignalStrategy class and assign parameters
    PSO_Strategy = strategies.CombinedSignalStrategy
    w, buy_threshold, sell_threshold = func_utils.get_split_w_threshold(best_pos)
    PSO_Strategy.w = w
    PSO_Strategy.buy_threshold = buy_threshold
    PSO_Strategy.sell_threshold = sell_threshold
    PSO_Strategy.period_list = gen_representation.period_list
    PSO_Strategy.moving_average_rules = gen_representation.moving_average_rules
    PSO_Strategy.moving_averages = gen_representation.moving_averages_test
    PSO_Strategy.optimizer = optimizer
    PSO_Strategy.gen_representation = gen_representation

    df_test = gen_representation.df_test
    df_train = gen_representation.df_train

    PSO_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(PSO_Strategy, df_test, commission)

    # Guardamos los resultados
    execution_analysis.printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'particle_swarm_optimization')
    # Guardamos la grafica de la simulacion
    execution_plot.plot_simulation(PSO_Cerebro, 'particle_swarm_optimization', data_name, s_test, e_test)

    return PSO_Cerebro, PSO_Strategy
