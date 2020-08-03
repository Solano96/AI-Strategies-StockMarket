# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys, getopt
from datetime import datetime, timedelta

import utils.func_utils as func_utils

from strategies_execution.executions import execute_buy_and_hold_strategy
from strategies_execution.executions import execute_classic_strategy
from strategies_execution.executions import execute_neural_network_strategy
from strategies_execution.executions import execute_pso_strategy

import strategies_execution.execution_plot as execution_plot

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


def main(argv):
    strategy = ''
    quote = ''
    commission = 0.001

    s_train, e_train = '2009-12-22', '2011-12-21'
    s_test, e_test = '2011-12-22', '2013-12-22'

    try:
        opts, args = getopt.getopt(argv, 'hs:q:f:t:', ['help', 'strategy=', 'quote=', 'from-date=', 'to-date=',
                                                       'nn-gain=', 'nn-loss=', 'nn-days=', 'nn-epochs=',
                                                       'pso-normalization='])
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

        for opt, arg in opts:
            if opt in ("--pso-normalization"):
                normalization = arg

        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        PSO_Cerebro, PSO_Strategy = execute_pso_strategy(df, options, commission, quote, s_test, e_test, normalization)
        strategy_list.append((PSO_Strategy, 'Particle Swarm Optimization'))

    execution_plot.plot_capital(strategy_list, quote, strategy, s_test, e_test)


if __name__ == "__main__":
    main(sys.argv[1:])
