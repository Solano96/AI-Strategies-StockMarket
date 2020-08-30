# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys, getopt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler

import src.utils.func_utils as func_utils

# Import classes
from src.classes.myCerebro import MyCerebro
from src.classes.myAnalyzer import MyAnalyzer
from src.classes.myBuySell import MyBuySell
from src.classes.maxRiskSizer import MaxRiskSizer

import src.classes.model as model
import src.classes.geneticRepresentation as geneticRepresentation

# Import strategies execution
import src.strategies_execution.execution_analysis as execution_analysis
import src.strategies_execution.execution_plot as execution_plot

# Import strategies
from src.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from src.strategies.classic_strategy import ClassicStrategy
from src.strategies.neural_network_strategy import NeuralNetworkStrategy
from src.strategies.combined_signal_strategy import CombinedSignalStrategy
from src.strategies.one_moving_average_strategy import OneMovingAverageStrategy
from src.strategies.moving_averages_cross_strategy import MovingAveragesCrossStrategy

import pyswarms as ps

import backtrader as bt
import backtrader.plot
import matplotlib
import matplotlib.pyplot as plt

from numpy.random import seed

seed(1)
# Opciones de ejecucion
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=sys.maxsize)


def print_execution_name(execution_name):
    print("\n --------------- ", execution_name, " --------------- \n")


def execute_strategy(strategy, df, commission, info, training_params=None, **kwargs):
    """
    Execute strategy on data history contained in df
    :param strategy: buying and selling strategy to be used
    :param df: dataframe with historical data
    :param commission: commission to be paid on each operation
    :returns:
        - cerebro - execution engine
    """

    # Create cerebro instance
    cerebro = MyCerebro()

    # Add strategy to cerebro
    strategy_index = cerebro.addstrategy(strategy, **kwargs)

    # Feed cerebro with historical data
    data = bt.feeds.PandasData(dataname = df)
    cerebro.adddata(data)

    # Add sizer
    cerebro.addsizer(MaxRiskSizer, risk=1.0)

    # Add analyzers to cerebro
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawDown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeAnalyzer")
    cerebro.addanalyzer(MyAnalyzer, _name = "myAnalyzer")

    # Change buy sell observer
    bt.observers.BuySell = MyBuySell

    # Set initial cash and commision
    cerebro.broker.setcash(6000.0)
    cerebro.broker.setcommission(commission=commission)

    initial_value = cerebro.broker.getvalue()

    print('\nValor inicial de la cartera: %.2f' % initial_value)

    # Execute cerebro
    strats = cerebro.run()

    final_value = cerebro.broker.getvalue()

    print('Valor final de la cartera  : %.2f' % final_value)

    # Get analysis from analyzers
    dd = strats[0].analyzers.drawDown.get_analysis()
    ta = strats[0].analyzers.tradeAnalyzer.get_analysis()
    ma = strats[0].analyzers.myAnalyzer.get_analysis()

    # Get analysis from analyzers
    drawDownAnalyzer = strats[0].analyzers.drawDown.get_analysis()
    tradeAnalyzer = strats[0].analyzers.tradeAnalyzer.get_analysis()
    myAnalyzer = strats[0].analyzers.myAnalyzer.get_analysis()

    avg_profit_trade = round(myAnalyzer.avg.profit_trade,2)
    avg_loss_trade = round(myAnalyzer.avg.loss_trade,2)

    avg_profit_loss = 'NaN'

    if avg_loss_trade != 0:
        avg_profit_loss = round((-1)*avg_profit_trade/avg_loss_trade,2)

    metrics = {
        'Inicial': initial_value,
        'Final': final_value,
        'Ganancia(%)': (final_value-initial_value)/initial_value,
        'Ganancias': round(final_value-initial_value,2),
        'Max DD': round((-1.0)*drawDownAnalyzer.max.drawdown,2),
        'Trades total': int(myAnalyzer.trades.total),
        'Trades+': int(myAnalyzer.trades.positives),
        'Trades-': int(myAnalyzer.trades.negatives),
        'Avg trade': round(myAnalyzer.avg.trade,2),
        'Avg profit': avg_profit_trade,
        'Avg loss': avg_loss_trade,
        'Profit/Loss': avg_profit_loss
    }

    params = kwargs

    if len(params) == 0:
        params = dict(strategy.params._getitems())

    execution_analysis.printAnalysis(info, params, metrics, training_params)
    execution_analysis.printAnalysisPDF(cerebro, info, params, metrics, training_params)

    return cerebro


def optimize_strategy(df, commission, strategy, to_date, **kwargs):
    """
    Get best params for a given strategy
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param strategy: buying and selling strategy to be used
    :param to_date: simulation final date
    :return: params with higher profit
    """

    s_test_date = datetime.strptime(to_date, '%Y-%m-%d')
    start_train = s_test_date.replace(year = s_test_date.year - 2)
    end_train = s_test_date - timedelta(days=1)

    df_train = df[start_train:end_train]

    # Create cerebro instance
    cerebro = bt.Cerebro()
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returnAnalyzer")

    # Add strategy to cerebro
    train_strategy = strategy
    train_strategy.printlog = False
    strats = cerebro.optstrategy(train_strategy, **kwargs)

    # Feed cerebro with historical data
    data = bt.feeds.PandasData(dataname = df_train)
    cerebro.adddata(data)

    # Add sizer
    cerebro.addsizer(MaxRiskSizer, risk=1.0)

    # Set initial cash and commision
    cerebro.broker.setcash(6000.0)
    cerebro.broker.setcommission(commission=commission)

    # Execute cerebro
    stratruns = cerebro.run()

    best_parameters = dict()
    best_value = 0

    # Search best parameters
    for stratrun in stratruns:
        for strat in stratrun:
            final_value = strat.analyzers[0]._value_end

            if best_value < final_value:
                best_value = final_value
                best_parameters = strat.p._getkwargs()

    return best_parameters


def execute_buy_and_hold_strategy(df, commission, data_name, start_date, end_date):
    """
    Execute buy and hold strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - BH_Cerebro - execution engine
        - BH_Strategy - buy and hold strategy instance
    """

    print_execution_name("Estrategia: comprar y mantener")

    strategy_name = 'comprar_y_mantener'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    df = df[start_date:end_date]

    BH_Strategy =  BuyAndHoldStrategy
    BH_Cerebro = execute_strategy(BH_Strategy, df, commission, info)

    # Save simulation chart
    execution_plot.plot_simulation(BH_Cerebro, strategy_name, data_name, start_date, end_date)

    return BH_Cerebro, BH_Strategy


def execute_classic_strategy(df, commission, data_name, start_date, end_date):
    """
    Execute classic strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - Classic_Cerebro - execution engine
        - Classic_Strategy - classic strategy instance
    """

    print_execution_name("Estrategia: clásica")

    strategy_name = 'estrategia_clasica'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    df = df[start_date:end_date]

    Classic_Strategy =  ClassicStrategy
    Classic_Cerebro = execute_strategy(Classic_Strategy, df, commission, info)

    # Save simulation chart
    execution_plot.plot_simulation(Classic_Cerebro, strategy_name, data_name, start_date, end_date)

    return Classic_Cerebro, Classic_Strategy


def execute_one_moving_average_strategy(df, commission, data_name, start_date, end_date):
    """
    Execute one moving average strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - OMA_Cerebro - execution engine
        - OMA_Strategy - one moving average strategy instance
    """

    print_execution_name("Estrategia: media móvil")

    strategy_name = 'estrategia_media_movil'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    params = {'maperiod': range(5, 50)}

    # Get best params in past period
    best_parameters = optimize_strategy(df, commission, OneMovingAverageStrategy, start_date, **params)

    df = df[start_date:end_date]

    OMA_Strategy =  OneMovingAverageStrategy
    OMA_Cerebro = execute_strategy(OMA_Strategy, df, commission, info, **best_parameters)

    # Save simulation chart
    execution_plot.plot_simulation(OMA_Cerebro, strategy_name, data_name, start_date, end_date)

    return OMA_Cerebro, OMA_Strategy


def execute_moving_averages_cross_strategy(df, commission, data_name, start_date, end_date, optimize=False, **kwargs):
    """
    Execute moving averages cross strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :param optimize: if True then optimize strategy
    :return:
        - MAC_Cerebro - execution engine
        - MAC_Strategy - moving averages cross strategy instance
    """

    print_execution_name("Estrategia: cruce de medias móviles")

    strategy_name = 'estrategia_cruce_medias_moviles'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    if optimize:
        print('Optimizando (esto puede tardar)...')

        # Range of values to optimize
        params = {
            'ma_short': range(5, 30),
            'ma_long': range(14, 60)
        }

        # Get best params in past period
        kwargs = optimize_strategy(df, commission, MovingAveragesCrossStrategy, start_date, **params)

    df = df[start_date:end_date]

    MAC_Strategy =  MovingAveragesCrossStrategy
    MAC_Cerebro = execute_strategy(MAC_Strategy, df, commission, info, **kwargs)

    # Save simulation chart
    execution_plot.plot_simulation(MAC_Cerebro, strategy_name, data_name, start_date, end_date)

    return MAC_Cerebro, MAC_Strategy


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
    neural_network = model.NeuralNetwork()
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

    # Save simulation chart
    execution_plot.plot_simulation(NN_Cerebro, 'red_neuronal', data_name, start_date, end_date)

    return NN_Cerebro, NN_Strategy


def execute_pso_strategy(df, options, commission, data_name, s_test, e_test, iters=100, normalization='exponential'):
    """
    Execute particle swarm optimization strategy on data history contained in df
    :param df: dataframe with historical data
    :param options: dict with the following parameters
        - c1 - cognitive parameter with which the particle follows its personal best
        - c2 - social parameter with which the particle follows the swarm's global best position
        - w - parameter that controls the inertia of the swarm's movement
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - PSO_Cerebro - execution engine
        - PSO_Strategy - pso strategy instance
    """

    print_execution_name("Estrategia: particle swar optimization")

    # ------------ Obtenemos los conjuntos de train y test ------------ #

    s_test_date = datetime.strptime(s_test, '%Y-%m-%d')
    s_train = s_test_date.replace(year = s_test_date.year - 2)
    #s_train = s_test_date - timedelta(days=180)
    e_train = s_test_date - timedelta(days=1)

    gen_representation = geneticRepresentation.GeneticRepresentation(df, s_train, e_train, s_test, e_test)

    # ------------ Fijamos hiperparámetros ------------ #

    n_particles=50
    dimensions=len(gen_representation.moving_average_rules)+2

    if normalization == 'exponential':
        max_bound = 1.0 * np.ones(dimensions-2)
        min_bound = -max_bound
    elif normalization == 'l1':
        max_bound = 1.0 * np.ones(dimensions-2)
        min_bound = np.zeros(dimensions-2)

    max_bound = np.append(max_bound, [1.0, 0.0])
    min_bound = np.append(min_bound, [0.0, -1.0])
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)

    # Perform optimization
    kwargs={'from_date': s_train, 'to_date': e_train}
    best_cost, best_pos = optimizer.optimize(gen_representation.cost_function, iters=iters, **kwargs)

    # Create an instance from CombinedSignalStrategy class and assign parameters
    PSO_Strategy = CombinedSignalStrategy
    w, buy_threshold, sell_threshold = func_utils.get_split_w_threshold(best_pos)

    PSO_Strategy.w = w
    PSO_Strategy.buy_threshold = buy_threshold
    PSO_Strategy.sell_threshold = sell_threshold
    PSO_Strategy.period_list = gen_representation.period_list
    PSO_Strategy.moving_average_rules = gen_representation.moving_average_rules
    PSO_Strategy.moving_averages = gen_representation.moving_averages_test
    PSO_Strategy.optimizer = optimizer
    PSO_Strategy.gen_representation = gen_representation
    PSO_Strategy.normalization = normalization

    df_test = gen_representation.df_test
    df_train = gen_representation.df_train

    PSO_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(PSO_Strategy, df_test, commission)

    # Guardamos los resultados
    strategy_name = 'particle_swarm_optimization'
    execution_analysis.printAnalysis(strategy_name, data_name, initial_value, final_value, ta, dd, ma)
    execution_analysis.printAnalysisPDF(PSO_Cerebro, strategy_name, data_name, initial_value, final_value, ta, dd, ma, s_test, e_test)

    # Guardamos la grafica de la simulacion
    execution_plot.plot_simulation(PSO_Cerebro, 'particle_swarm_optimization', data_name, s_test, e_test)

    return PSO_Cerebro, PSO_Strategy
