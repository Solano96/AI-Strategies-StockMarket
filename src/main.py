# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys, getopt

from sklearn.preprocessing import StandardScaler

import utils.func_utils as func_utils
import utils.myCerebro as myCerebro
import utils.myAnalyzer as myAnalyzer
import utils.testStrategy as testStrategy
import utils.model as model
import utils.strategies as strategies
import utils.geneticRepresentation as geneticRepresentation

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


def printAnalysis(data_name, initial, final, tradeAnalyzer, drawDownAnalyzer, myAnalyzer, file_name, train_accuracy=None, test_accuracy=None):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''

    f = open ('../resultados/resultados_' + file_name + '.txt','a')
    f.write(data_name)
    f.write("\n\n")

    if train_accuracy != None and test_accuracy != None:
        f.write("Train score : %.2f\n" % train_accuracy)
        f.write("Test score  : %.2f\n\n" % test_accuracy)

    percentage_profit = (final-initial)/initial

    f.write("Inicial     : %.2f\n" % initial)
    f.write("Final       : %.2f\n" % final)
    f.write("Ganancia(%%) : %.2f\n" % percentage_profit)

    net_profit = round(final-initial,2)
    maxdd = round((-1.0)*drawDownAnalyzer.max.drawdown,2)
    trades_total = int(myAnalyzer.trades.total)
    trades_positives = int(myAnalyzer.trades.positives)
    trades_negatives = int(myAnalyzer.trades.negatives)
    avg_trade = round(myAnalyzer.avg.trade,2)
    avg_profit_trade = round(myAnalyzer.avg.profit_trade,2)
    avg_loss_trade = round(myAnalyzer.avg.loss_trade,2)

    avg_profit_loss = 99999999

    if avg_loss_trade != 0:
        avg_profit_loss = round((-1)*avg_profit_trade/avg_loss_trade,2)

    f.write("Ganancias   : %.2f\n" % net_profit)
    f.write("Max DD      : %.2f\n" % maxdd)
    f.write("Trades total: %i\n" % trades_total)
    f.write("Trades+     : %i\n" % trades_positives)
    f.write("Trades-     : %i\n" % trades_negatives)
    f.write("Avg trade   : %.2f\n" % avg_trade)
    f.write("Avg profit  : %.2f\n" % avg_profit_trade)
    f.write("Avg loss    : %.2f\n" % avg_loss_trade)
    f.write("Profit/Loss : %.2f\n\n\n" % avg_profit_loss)

    f.close()


def execute_strategy(strategy, df, commission):
    # Creamos la instancia cerebro
    cerebro = myCerebro.MyCerebro()

    # Añadimos la estrategia al cerebro
    cerebro.addstrategy(strategy)

    # Añadimos los datos al cerebro
    data = bt.feeds.PandasData(dataname = df)
    cerebro.adddata(data)

    # Añadimos los analizadores
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawDown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeAnalyzer")
    cerebro.addanalyzer(myAnalyzer.MyAnalyzer, _name = "myAnalyzer")

    # Fijamos el dinero inicial y la comisión
    cerebro.broker.setcash(6000.0)
    cerebro.broker.setcommission(commission=commission)

    initial_value = cerebro.broker.getvalue()

    print('\nValor inicial de la cartera: %.2f' % initial_value)

    # Ejecutamos la estrategia sobre los datos del test
    strats = cerebro.run()

    final_value = cerebro.broker.getvalue()

    print('Valor final de la cartera  : %.2f' % final_value)

    # print the analyzers
    dd = strats[0].analyzers.drawDown.get_analysis()
    ta = strats[0].analyzers.tradeAnalyzer.get_analysis()
    ma = strats[0].analyzers.myAnalyzer.get_analysis()

    return cerebro, initial_value, final_value, ta, dd, ma


def plot_simulation(cerebro, file_name, data_name, from_date, to_date):

    cerebro.getFig(iplot=False)

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)

    fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(12, 6.46)

    #plt.show()

    if not os.path.exists('../img/simulacion_' + file_name):
        os.makedirs('../img/simulacion_' + file_name)

    plt.savefig('../img/simulacion_' + file_name + '/' + data_name + '_' + from_date + '_' + to_date + '_' + file_name + '.png')


def plot_capital(strategy_list, data_name, img_name, from_date, to_date):

    fig = plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111)

    for strategy, name_strategy in strategy_list:
        ax.plot(strategy.dates, strategy.values, label=name_strategy)

    ax.legend(loc='upper left')
    ax.yaxis.grid(linestyle="-")

    if not os.path.exists('../img/ganancias/'):
        os.makedirs('../img/ganancias/')

    plt.savefig('../img/ganancias/' + data_name + '_' + from_date + '_' + to_date + '_' + img_name + '.png')


def execute_neural_network_strategy(df, options, comm, data_name, s_train, e_train, s_test, e_test):

    print("\n ############### Estrategia: red neuronal ############### \n")

    # ------------ Get parameters ------------#
    gain = options['gain']
    loss = options['loss']
    n_day = options['n_day']
    epochs = options['epochs']

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
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'red_neuronal', train_accuracy, test_accuracy)
    # Save simulation chart
    plot_simulation(NN_Cerebro, 'red_neuronal', data_name, s_test, e_test)

    return NN_Cerebro, NN_Strategy


def execute_buy_and_hold_strategy(df, commission, data_name, start_date, end_date):

    print("\n ############### Estrategia: comprar y mantener ############### \n")

    df = df[start_date:end_date]

    BH_Strategy =  strategies.BuyAndHoldStrategy
    BH_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(BH_Strategy, df, commission)

    # Save results
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'comprar_y_mantener')
    # Save simulation chart
    plot_simulation(BH_Cerebro, 'comprar_y_mantener', data_name, start_date, end_date)

    return BH_Cerebro, BH_Strategy


def execute_classic_strategy(df, commission, data_name, start_date, end_date):

    print("\n ############### Estrategia: clásica ############### \n")

    df = df[start_date:end_date]

    Classic_Strategy =  strategies.ClassicStrategy
    Classic_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(Classic_Strategy, df, commission)

    # Save results
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'estrategia_clasica')
    # Save simulation chart
    plot_simulation(Classic_Cerebro, 'estrategia_clasica', data_name, start_date, end_date)

    return Classic_Cerebro, Classic_Strategy



def execute_pso_strategy(df, commission, data_name, s_train, e_train, s_test, e_test):

    print("\n ############### Estrategia: particle swar optimization ############### \n")

    # ------------ Obtenemos los conjuntos de train y test ------------ #

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
    best_cost, best_pos = optimizer.optimize(gen_representation.cost_function, iters=iters)

    # Create an instance from CombinedSignalStrategy class and assign parameters
    PSO_Strategy = strategies.CombinedSignalStrategy
    w, buy_threshold, sell_threshold = func_utils.get_split_w_threshold(best_pos)
    PSO_Strategy.w = w
    PSO_Strategy.buy_threshold = buy_threshold
    PSO_Strategy.sell_threshold = sell_threshold
    PSO_Strategy.period_list = gen_representation.period_list
    PSO_Strategy.moving_average_rules = gen_representation.moving_average_rules
    PSO_Strategy.moving_averages = gen_representation.moving_averages_test

    df_test = gen_representation.df_test
    df_train = gen_representation.df_train

    PSO_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(PSO_Strategy, df_test, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'particle_swarm_optimization')
    # Guardamos la grafica de la simulacion
    plot_simulation(PSO_Cerebro, 'particle_swarm_optimization', data_name, s_test, e_test)

    return PSO_Cerebro, PSO_Strategy

def main(argv):
    strategy = ''
    quote = ''
    commission = 0.001

    s_train, e_train = '2009-12-22', '2011-12-21'
    s_test, e_test = '2011-12-22', '2013-12-22'

    strategy_func_switcher = {
        'buy-and-hold':   (execute_buy_and_hold_strategy, 'Comprar y Mantener'),
        'classic':        (execute_classic_strategy, 'Estrategia Clásica'),
        'neural-network': (execute_neural_network_strategy, 'Red Neuronal'),
        'combined-pso':   (execute_pso_strategy, 'Particle Swarm Optimization')
    }

    try:
        opts, args = getopt.getopt(argv, 'hs:q:f:t:', ['help', 'strategy=', 'quote=', 'from-date=', 'to-date='])
    except getopt.GetoptError:
        print('main.py -s <strategy> -q <quote> -f <from-date> -t <to-date>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('\nDESCRIPTION')
            print('\n\tThis app allow you to test different strategies for investing in the stock market.')
            print('\nUSAGE')
            print('\n\tmain.py -s <strategy> -q <quote>')
            print('\nOPTIONS')
            print('\n\t-s, --strategy\tSelect a strategy between: buy-and-hold | classic | neural-network | combined-signal-pso | all.')
            print('\n\t-q, --quote\tUse as quote any market abbreviation recognized by yahoo finance. Examples: AAPL | FB | GOOGL | AMZN | ...')
            print('\n\t-f, --from-date\tStart date in simulation.')
            print('\n\t-t, --to-date\tEnd date in simulation.')
            print('\n\t-h, --help\tDisplay help.')
            sys.exit()
        elif opt in ("-s", "--strategy"):
            strategy = arg
        elif opt in ("-q", "--quote"):
            quote = arg
        elif opt in ("-f", "--from-date"):
            s_test = arg
        elif opt in ("-t", "--to-date"):
            e_test = arg

    df = func_utils.getData(quote)

    strategy_list = []

    # Execute buy and hold strategy
    if strategy in ('buy-and-hold', 'all'):
        BH_Cerebro, BH_Strategy = execute_buy_and_hold_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((BH_Strategy, 'Comprar y Mantener'))

    # Execute classic strategy
    if strategy in ('classic', 'all'):
        Classic_Cerebro, Classic_Strategy = execute_classic_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((Classic_Strategy, 'Estrategia Clásica'))

    # Execute neural network strategy
    if strategy in ('neural-network', 'all'):
        options = {'gain': 0.07, 'loss': 0.03, 'n_day': 10, 'epochs': 300}
        NN_Cerebro, NN_Strategy = execute_neural_network_strategy(df, options, commission, quote, s_train, e_train, s_test, e_test)
        strategy_list.append((NN_Strategy, 'Red Neuronal'))

    # Execute combined signal strategy optimized with pso
    if strategy in ('combined-signal-pso', 'all'):
        PSO_Cerebro, PSO_Strategy = execute_pso_strategy(df, commission, quote, s_train, e_train, s_test, e_test)
        strategy_list.append((PSO_Strategy, 'Particle Swarm Optimization'))

    plot_capital(strategy_list, quote, strategy, s_test, e_test)


if __name__ == "__main__":
    main(sys.argv[1:])
