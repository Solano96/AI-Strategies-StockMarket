# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys
# Just disables the warning, doesn't enable AVX/FMA
import os

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


def plot_simulation(cerebro, file_name):

    cerebro.getFig(iplot=False)

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)

    fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(12, 6.46)

    #plt.show()

    if not os.path.exists('../img/simulacion_' + file_name):
        os.makedirs('../img/simulacion_' + file_name)

    plt.savefig('../img/simulacion_' + file_name + '/' + data_name + '_' + file_name + '.png')


def plot_capital(strategy_list, data_name):

    fig = plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111)

    for strategy, name_strategy in strategy_list:
        ax.plot(strategy.dates, strategy.values, label=name_strategy)

    ax.legend(loc='upper left')
    ax.yaxis.grid(linestyle="-")

    if not os.path.exists('../img/ganancias/'):
        os.makedirs('../img/ganancias/')

    plt.savefig('../img/ganancias/' + data_name + '.png')


def execute_neural_network_strategy(df, gain, loss, n_day , comm, data_name, s_train, e_train, s_test, e_test):

    print("\n ############### Estrategia: red neuronal ############### \n")

    # ------------ Preparamos los datos ------------ #

    df = func_utils.add_features(df)
    df = func_utils.add_label(df, gain = gain, loss = loss, n_day = n_day, commission = comm)

    # ------------ Obtenemos los conjuntos de train y test ------------ #

    df_train, df_test, X_train, X_test, y_train, y_test = func_utils.split_df_date(df, s_train, e_train, s_test, e_test)

    # ------------ Normalizamos los datos ------------ #

    print("Normalizando datos...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform (X_test)

    # Ponemos los datos en formato correcto para usarlos en keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # ------------ Obtenemos el modelo de predicción ------------ #

    print("Entrenando red neuronal...")

    neural_network = model.NeuralNetwork()
    neural_network.build_model(input_shape = (X_train.shape[1], 1))
    neural_network.train(X_train, y_train, epochs = epochs)

    # ------------ Porcentaje de acierto en la predicción ------------ #

    train_accuracy = neural_network.get_accuracy(X_train, y_train)
    test_accuracy = neural_network.get_accuracy(X_test, y_test)

    print("\nRESULTADOS PREDICCION:\n")
    print("TRAIN :: Porcentaje de acierto: " + str(train_accuracy))
    print("TEST  :: Porcentaje de acierto: " + str(test_accuracy))

    # ------------------------ Backtesting ------------------------ #

    # Inicializamos la memoria de la red neuronal
    neural_network.init_memory(X_train[len(X_train)-15:len(X_train)], y_train[len(y_train)-15:len(y_train)])

    # Creamos una instancia de la clase NeuralNetworkStrategy
    NN_Strategy = strategies.NeuralNetworkStrategy
    NN_Strategy.X_test = X_test
    NN_Strategy.y_test = y_test
    NN_Strategy.model = neural_network
    NN_Strategy.n_day = n_day

    # Ejecutamos la estrategia
    NN_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(NN_Strategy, df_test, comm)
    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'red_neuronal', train_accuracy, test_accuracy)
    # Guardamos la grafica de la simulacion
    plot_simulation(NN_Cerebro, 'red_neuronal')

    return NN_Cerebro, NN_Strategy


def execute_buy_and_hold_strategy(df, commission, data_name, start_date, end_date):

    print("\n ############### Estrategia: comprar y mantener ############### \n")

    df = df[start_date:end_date]

    BH_Strategy =  strategies.BuyAndHoldStrategy
    BH_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(BH_Strategy, df, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'comprar_y_mantener')
    # Guardamos la grafica de la simulacion
    plot_simulation(BH_Cerebro, 'comprar_y_mantener')

    return BH_Cerebro, BH_Strategy


def execute_classic_strategy(df, commission, data_name, start_date, end_date):

    print("\n ############### Estrategia: clásica ############### \n")

    df = df[start_date:end_date]

    Classic_Strategy =  strategies.ClassicStrategy
    Classic_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(Classic_Strategy, df, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'estrategia_clasica')
    # Guardamos la grafica de la simulacion
    plot_simulation(Classic_Cerebro, 'estrategia_clasica')

    return Classic_Cerebro, Classic_Strategy



def execute_pso_strategy(df, commission, data_name, s_train, e_train, s_test, e_test):

    print("\n ############### Estrategia: particle swar optimization ############### \n")

    # ------------ Obtenemos los conjuntos de train y test ------------ #

    gen_representation = geneticRepresentation.GeneticRepresentation(df, s_train, e_train, s_test, e_test)

    # ------------ Fijamos hiperparámetros ------------ #

    n_particles=20
    dimensions=105
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    max_bound = 1.0 * np.ones(dimensions)
    #min_bound = -max_bound
    min_bound = np.zeros(dimensions)
    bounds = (min_bound, max_bound)
    iters = 100

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(gen_representation.cost_function, iters=iters)

    PSO_Strategy = strategies.CombinedSignalStrategy
    PSO_Strategy.w = best_pos/np.sum(best_pos)
    #PSO_Strategy.w = np.exp(best_pos)/np.sum(np.exp(best_pos))
    #PSO_Strategy.w = best_pos
    PSO_Strategy.period_list = gen_representation.period_list
    PSO_Strategy.moving_average_rules = gen_representation.moving_average_rules
    PSO_Strategy.moving_averages = gen_representation.moving_averages_test

    df_test = gen_representation.df_test
    df_train = gen_representation.df_train

    PSO_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(PSO_Strategy, df_test, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'particle_swarm_optimization')
    # Guardamos la grafica de la simulacion
    plot_simulation(PSO_Cerebro, 'particle_swarm_optimization')

    return PSO_Cerebro, PSO_Strategy


if __name__ == '__main__':

    # ----- Comprobamos que el modo de uso sea el correcto ----- #

    if len(sys.argv) != 2 and len(sys.argv) != 6:
        print("Error: modo de uso incorrecto.")
        print("Modo de uso 1: python main.py data_name")
        print("Modo de uso 2: python main.py data_name gain loss n_day epochs")
        exit()

    # Obtenemos el nombre de la base de datos
    data_name = sys.argv[1]
    print(data_name, "\n")

    # ------- Fijamos parametros ------- #

    commission = 0.001

    if len(sys.argv) == 2:
        gain = 0.1
        loss = 0.05
        n_day = 20
        epochs = 100
    else:
        gain = float(sys.argv[2])
        loss = float(sys.argv[3])
        n_day = int(sys.argv[4])
        epochs = int(sys.argv[5])

    # ------------ Obtenemos los datos ------------ #

    df = func_utils.getData(data_name)

    s_train, e_train = '2009-12-22', '2011-12-21'
    s_test, e_test = '2011-12-22', '2013-12-22'

    #s_train_genetic = '2011-08-01'
    #s_train, e_train = '2012-08-01', '2014-08-01'
    #s_test, e_test = '2014-08-01', '2016-08-01'

    #s_train, e_train = '2015-06-01', '2017-06-01'
    #s_test, e_test = '2017-06-01', '2019-06-01'

    BH_Cerebro, BH_Strategy = execute_buy_and_hold_strategy(df, commission, data_name, s_test, e_test)
    Classic_Cerebro, Classic_Strategy = execute_classic_strategy(df, commission, data_name, s_test, e_test)
    NN_Cerebro, NN_Strategy = execute_neural_network_strategy(df, gain, loss, n_day, commission, data_name, s_train, e_train, s_test, e_test)
    PSO_Cerebro, PSO_Strategy = execute_pso_strategy(df, commission, data_name, s_train, e_train, s_test, e_test)

    strategy_list = []
    strategy_list.append((BH_Strategy, 'Comprar y Mantener'))
    strategy_list.append((Classic_Strategy, 'Estrategia Clásica'))
    strategy_list.append((NN_Strategy, 'Red Neuronal'))
    strategy_list.append((PSO_Strategy, 'Particle Swarm Optimization'))

    plot_capital(strategy_list, data_name)
