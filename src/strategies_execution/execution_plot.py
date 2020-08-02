# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys, getopt
from datetime import datetime, timedelta


import utils.func_utils as func_utils
import utils.myCerebro as myCerebro
import utils.myAnalyzer as myAnalyzer
import utils.testStrategy as testStrategy
import utils.strategies as strategies

import backtrader as bt
import backtrader.plot
import matplotlib
import matplotlib.pyplot as plt

from numpy.random import seed

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Opciones de ejecucion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=sys.maxsize)


def plot_simulation(cerebro, file_name, data_name, from_date=None, to_date=None):
    """
    Plot strategy simulation
    :param cerebro:
    :param file_name: file name for the generated image
    :param data_name: quote data name
    :param from_date: start date of simulation
    :param to_date: end date of simulation
    """
    cerebro.getFig(iplot=False)

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)

    fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(12, 6.46)

    #plt.show()

    if not os.path.exists('../img/simulacion_' + file_name):
        os.makedirs('../img/simulacion_' + file_name)

    if from_date==None or to_date==None:
        plt.savefig('../img/simulacion_' + file_name + '/' + data_name + '_' + file_name + '.png')
    else:
        plt.savefig('../img/simulacion_' + file_name + '/' + data_name + '_' + from_date + '_' + to_date + '_' + file_name + '.png')


def plot_capital(strategy_list, data_name, img_name, from_date=None, to_date=None):
    """
    Plot chart with the capital of the strategy list
    :param strategy_list: list with the strategies and their respective names
    :param data_name: quote data name
    :param img_name: file name for the generated image
    :param from_date: start date of simulation
    :param to_date: end date of simulation
    """
    fig = plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111)

    for strategy, name_strategy in strategy_list:
        ax.plot(strategy.dates, strategy.values, label=name_strategy)

    ax.legend(loc='upper left')
    ax.yaxis.grid(linestyle="-")

    if not os.path.exists('../img/ganancias/'):
        os.makedirs('../img/ganancias/')

    if from_date==None or to_date==None:
        plt.savefig('../img/ganancias/' + data_name + '_' + img_name + '.png')
    else:
        plt.savefig('../img/ganancias/' + data_name + '_' + from_date + '_' + to_date + '_' + img_name + '.png')
