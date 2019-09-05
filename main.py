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

    f = open ('./resultados/resultados_' + file_name + '.txt','a')
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

    if not os.path.exists('img/simulacion_' + file_name):
        os.makedirs('img/simulacion_' + file_name)

    plt.savefig('img/simulacion_' + file_name + '/' + data_name + '_' + file_name + '.png')



def plot_capital(NN_Strategy, BH_Strategy, Classic_Strategy, data_name):

    fig = plt.figure()

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)

    ax = fig.add_subplot(111)
    ax.plot(NN_Strategy.dates, NN_Strategy.values, label='Red Neuronal')
    ax.plot(BH_Strategy.dates, BH_Strategy.values, label='Comprar y Mantener')
    ax.plot(Classic_Strategy.dates, Classic_Strategy.values, label='Estrategia Clásica')
    ax.legend(loc='upper left')
    ax.yaxis.grid(linestyle="-")

    #plt.show()

    if not os.path.exists('img/ganancias/'):
        os.makedirs('img/ganancias/')

    plt.savefig('img/ganancias/' + data_name + '.png')


def execute_neural_network_strategy(df, gain, loss, n_day , comm, data_name, s_train, e_train, s_test, e_test):

    print("\nEstrategia: red neuronal")

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

    print("\nEstrategia: comprar y mantener")

    df = df[start_date:end_date]

    BH_Strategy =  strategies.BuyAndHoldStrategy

    BH_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(BH_Strategy, df, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'comprar_y_mantener')

    # Guardamos la grafica de la simulacion
    plot_simulation(BH_Cerebro, 'comprar_y_mantener')

    return BH_Cerebro, BH_Strategy


def execute_classic_strategy(df, commission, data_name, start_date, end_date):

    print("\nEstrategia: clásica")

    df = df[start_date:end_date]

    Classic_Strategy =  strategies.ClassicStrategy

    Classic_Cerebro, initial_value, final_value, ta, dd, ma = execute_strategy(Classic_Strategy, df, commission)

    # Guardamos los resultados
    printAnalysis(data_name, initial_value, final_value, ta, dd, ma, 'estrategia_clasica')

    # Guardamos la grafica de la simulacion
    plot_simulation(Classic_Cerebro, 'estrategia_clasica')

    return Classic_Cerebro, Classic_Strategy



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

    #s_train, e_train = '2009-12-22', '2011-12-21'
    #s_test, e_test = '2011-12-22', '2013-12-22'

    s_train, e_train = '2012-08-01', '2014-08-01'
    s_test, e_test = '2014-08-01', '2016-08-01'

    #s_train, e_train = '2015-06-01', '2017-06-01'
    #s_test, e_test = '2017-06-01', '2019-06-01'

    NN_Cerebro, NN_Strategy = execute_neural_network_strategy(df, gain, loss, n_day, commission, data_name, s_train, e_train, s_test, e_test)
    BH_Cerebro, BH_Strategy = execute_buy_and_hold_strategy(df, commission, data_name, s_test, e_test)
    Classic_Cerebro, Classic_Strategy = execute_classic_strategy(df, commission, data_name, s_test, e_test)

    plot_capital(NN_Strategy, BH_Strategy, Classic_Strategy, data_name)

    exit()
    all_predictions = np.array(NN_Strategy.all_predictions)

    # plot values
    plt.title(data_name)
    plt.subplot(2, 1, 2)
    plt.plot(NN_Strategy.dates,all_predictions)
    plt.grid()
    plt.ylabel('Predictions')

    # plot closes
    plt.subplot(2, 1, 1)
    plt.plot(NN_Strategy.dates,NN_Strategy.closes)
    plt.ylabel('Precios cierre')

    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()



    # plot values
    plt.title(data_name)
    plt.subplot(2, 1, 1)
    plt.plot(ts.dates,ts.values)
    plt.ylabel('Capital')

    # plot closes
    plt.subplot(2, 1, 2)
    plt.plot(ts.dates,ts.closes)
    plt.ylabel('Precios cierre')

    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()

    #plt.savefig('img/ganancia_red_neuronal/g_' + data_name + '_' + str(gain) + '_' + str(loss) + '_' + str(n_day) + '_' + str(epochs) + '.png')



    comprar = np.ma.masked_where(np.array(df_test['label'])==1, df_test['Close'])
    vender = np.ma.masked_where(np.array(df_test['label'])==0, df_test['Close'])

    fig, ax = plt.subplots()
    ax.plot(df_test.index, comprar, 'r', df_test.index, vender, 'g')
    plt.show()

    all_predictions = np.array(ts.all_predictions)

    # plot values
    plt.title(data_name)
    plt.subplot(2, 1, 2)
    plt.plot(ts.dates,all_predictions)
    plt.grid()
    plt.ylabel('Predictions')

    # plot closes
    plt.subplot(2, 1, 1)
    plt.plot(ts.dates,ts.closes)
    plt.ylabel('Precios cierre')

    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()


    all_predictions = (all_predictions > 0.5)
    comprar = np.ma.masked_where(all_predictions==1, df_test['Close'])
    vender = np.ma.masked_where(all_predictions==0, df_test['Close'])

    fig, ax = plt.subplots()
    ax.plot(df_test.index, comprar, 'r', df_test.index, vender, 'g')
    plt.show()
