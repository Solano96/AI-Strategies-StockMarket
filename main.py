# -*- coding: utf-8 -*-
import logging
logging.disable(logging.CRITICAL)

# Just disables the warning, doesn't enable AVX/FMA
import os
import sys, getopt

import src.utils.func_utils as func_utils
from src.strategies_execution.executions import *
import src.strategies_execution.execution_plot as execution_plot

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def main(argv):
    strategy = ''
    quote = ''
    commission = 0.001

    s_train, e_train = '2009-12-22', '2011-12-21'
    s_test, e_test = '2011-12-22', '2013-12-22'

    optimize = False


    try:
        opts, args = getopt.getopt(argv, 'hs:q:f:t:vo',
                          ['help', 'strategy=', 'quote=', 'from-date=', 'to-date=',
                           # Neural network parameters
                           'nn-gain=', 'nn-loss=', 'nn-days=', 'nn-epochs=',
                           # PSO parameters
                           'pso-normalization=', 'pso-c1=', 'pso-c2=', 'pso-inertia=', 'pso-iters=',
                           'pso-retrain-repeat=', 'pso-retrain-interval=', 'pso-retrain-iters=',
                           # Two moving averages parameters
                           'ma-short=', 'ma-long=',
                           # One moving averages parameters
                           'ma-period=',
                           'optimize',
                           'verbose'])

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
            print('\n\t-s, --strategy\tSelect a strategy between:')
            print('\n\t\tbuy-and-hold: buy and hold until the end.')
            print('\n\t\tclassic: a classic strategy that use cross of moving averages and rsi indicator.')
            print('\n\t\tone-ma: cross of moving average with daily price.')
            print('\n\t\ttwo-ma: cross of two moving average short and long period.')
            print('\n\t\tneural-network: ')
            print('\n\t\tcombined-signal-pso:')
            print('\n\t\tall: execute all strategies.')
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
        elif opt in("-v", "--verbose"):
            logging.disable(logging.NOTSET)

    # Download data
    df = func_utils.getData(quote)

    strategy_list = []

    # -------------------- Execute buy and hold strategy -------------------- #

    if strategy in ('buy-and-hold', 'all'):
        BH_Cerebro, BH_Strategy = execute_buy_and_hold_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((BH_Strategy, 'Comprar y Mantener'))


    # -------------------- Execute classic strategy -------------------- #

    if strategy in ('classic'):
        Classic_Cerebro, Classic_Strategy = execute_classic_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((Classic_Strategy, 'Estrategia Clásica'))


    # -------------------- Execute one moving average -------------------- #

    if strategy in ('one-ma', 'all'):

        params = {}

        for opt, arg in opts:
            if opt == '--ma-period':
                print(opt)
                params['ma_period'] = int(arg)
            elif opt in ("-o", "--optimize"):
                optimize = True

        OMA_Cerebro, OMA_Strategy = execute_one_moving_average_strategy(df, commission, quote, s_test, e_test, optimize, **params)
        strategy_list.append((OMA_Strategy, 'Estrategia Media Móvil'))


    # -------------------- Execute two moving average -------------------- #

    if strategy in ('two-ma', 'all'):

        params = {}

        for opt, arg in opts:
            if opt == '--ma-short':
                print(opt)
                params['ma_short'] = int(arg)
            elif opt == '--ma-long':
                params['ma_long'] = int(arg)
            elif opt in ("-o", "--optimize"):
                optimize = True

        MAC_Cerebro, MAC_Strategy = execute_moving_averages_cross_strategy(df, commission, quote, s_test, e_test, optimize, **params)
        strategy_list.append((MAC_Strategy, 'Estrategia Cruce Medias Móviles'))


    # -------------------- Execute neural network strategy -------------------- #

    if strategy in ('neural-network'):

        options = {'gain': 0.07, 'loss': 0.05, 'n_day': 10, 'epochs': 300}

        for opt, arg in opts:
            if opt == "--nn-gain":
                options['gain'] = float(arg)
            elif opt == "--nn-loss":
                options['loss'] = float(arg)
            elif opt == "--nn-days":
                options['n_day'] = int(arg)
            elif opt == "--nn-epochs":
                options['epochs'] = int(arg)

        NN_Cerebro, NN_Strategy = execute_neural_network_strategy(df, options, commission, quote, s_test, e_test)
        strategy_list.append((NN_Strategy, 'Red Neuronal'))


    # -------------------- Execute combined signal strategy optimized with pso -------------------- #

    if strategy in ('combined-signal-pso', 'all'):

        normalization = 'exponential'

        options = {
            'c1': 0.5, # cognitive parameter
            'c2': 0.3, # social parameter
            'w': 0.9,  # inertia parameter
            'k': 10,   # number of neighbors to be considered
            'p': 2     # the Minkowski p-norm to use. 1: norma L1, 2: norma L2.
        }

        retrain_params = {
            'repeat': 90,
            'interval': 100,
            'iters': 10
        }

        iters = 400

        for opt, arg in opts:
            if opt in ("--pso-normalization"):
                normalization = arg
            elif opt in ("--pso-c1"):
                options['c1'] = float(arg)
            elif opt in ("--pso-c2"):
                options['c2'] = float(arg)
            elif opt in ("--pso-inertia"):
                options['w'] = float(arg)
            elif opt in ("--pso-iters"):
                iters = int(arg)
            elif opt in ("--pso-retrain-repeat"):
                retrain_params['repeat'] = int(arg)
            elif opt in ("--pso-retrain-interval"):
                retrain_params['interval'] = int(arg)
            elif opt in ("--pso-retrain-iters"):
                retrain_params['iters'] = int(arg)

        PSO_Cerebro, PSO_Strategy = execute_pso_strategy(df, options, retrain_params, commission, quote, s_test, e_test, iters, normalization)
        strategy_list.append((PSO_Strategy, 'Particle Swarm Optimization'))

    if len(strategy_list) == 0:
        print("ERROR: incorrect strategy name. Please select one between: buy-and-hold | classic | neural-network | combined-signal-pso | all.")
        sys.exit(2)

    execution_plot.plot_capital(strategy_list, quote, strategy, s_test, e_test)


if __name__ == "__main__":
    main(sys.argv[1:])


'''

EXECUTION EXAMPLES
----------------------------------------------------------

PSO Strategy

python3 main.py -s all -q SAN -f 2011-12-22 -t 2013-12-22 --pso-iters 10

python3 main.py -s all -q SAN -f 2011-12-22 -t 2013-12-22 --pso-iters 10 --pso-retrain-repeat 100 --pso-retrain-interval 50 --pso-retrain-iters 5





'''
