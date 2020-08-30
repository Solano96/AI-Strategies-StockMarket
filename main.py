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


    try:
        opts, args = getopt.getopt(argv, 'hs:q:f:t:v', ['help', 'strategy=', 'quote=', 'from-date=', 'to-date=',
                                                       'nn-gain=', 'nn-loss=', 'nn-days=', 'nn-epochs=',
                                                       'pso-normalization=', 'pso-c1=', 'pso-c2=', 'pso-inertia=', 'pso-iters=',
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

    # Execute one moving average
    if strategy in ('one-ma', 'all'):
        OMA_Cerebro, OMA_Strategy = execute_one_moving_average_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((OMA_Strategy, 'Estrategia Media Móvil'))

    # Execute two moving average
    if strategy in ('two-ma', 'all'):
        MAC_Cerebro, MAC_Strategy = execute_moving_averages_cross_strategy(df, commission, quote, s_test, e_test)
        strategy_list.append((MAC_Strategy, 'Estrategia Cruce Medias Móviles'))

    # Execute neural network strategy
    if strategy in ('neural-network', 'all'):

        options = {'gain': 0.07, 'loss': 0.05, 'n_day': 10, 'epochs': 300}

        for opt, arg in opts:
            if opt in ("--nn-gain"):
                options['gains'] = float(arg)
            elif opt in ("--nn-loss"):
                options['loss'] = float(arg)
            elif opt in ("--nn-days"):
                options['n_day'] = int(arg)
            elif opt in ("--nn-epochs"):
                options['epochs'] = int(arg)

        NN_Cerebro, NN_Strategy = execute_neural_network_strategy(df, options, commission, quote, s_test, e_test)
        strategy_list.append((NN_Strategy, 'Red Neuronal'))

    # Execute combined signal strategy optimized with pso
    if strategy in ('combined-signal-pso', 'all'):

        normalization = 'exponential'
        c1 = 0.5
        c2 = 0.3
        w = 0.9
        iters = 400

        for opt, arg in opts:
            if opt in ("--pso-normalization"):
                normalization = arg
            elif opt in ("--pso-c1"):
                c1 = arg
            elif opt in ("--pso-c2"):
                c2 = arg
            elif opt in ("--pso-inertia"):
                w = arg
            elif opt in ("--pso-iters"):
                iters = arg

        options = {'c1': c1, 'c2': c2, 'w': w}

        PSO_Cerebro, PSO_Strategy = execute_pso_strategy(df, options, commission, quote, s_test, e_test, iters, normalization)
        strategy_list.append((PSO_Strategy, 'Particle Swarm Optimization'))

    if len(strategy_list) == 0:
        print("ERROR: incorrect strategy name. Please select one between: buy-and-hold | classic | neural-network | combined-signal-pso | all.")
        sys.exit(2)

    execution_plot.plot_capital(strategy_list, quote, strategy, s_test, e_test)


if __name__ == "__main__":
    main(sys.argv[1:])
