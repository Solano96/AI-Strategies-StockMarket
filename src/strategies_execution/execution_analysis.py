# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys

import backtrader as bt

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def printAnalysis(file_name, data_name, initial_value, final_value, tradeAnalyzer, drawDownAnalyzer, myAnalyzer,
                  train_accuracy=None, test_accuracy=None):
    '''
    Function to print the Technical Analysis results in a nice format.
    :param file_name: file name to print the analysis
    :param data_name: quote data name
    :param initial_value: initial value of the portfolio
    :param final_value: final value of the portfolio
    :param tradeAnalyzer: trade analyzer instance
    :param drawDownAnalyzer: drawdown analyzer instance
    :param myAnalyzer: myAnalyzer instance
    :param train_accuracy: train accuracy (optional)
    :param test_accuracy: test accuracy (optional)
    '''

    f = open ('../resultados/resultados_' + file_name + '.txt','a')
    f.write(data_name)
    f.write("\n\n")

    if train_accuracy != None and test_accuracy != None:
        f.write("Train score : %.2f\n" % train_accuracy)
        f.write("Test score  : %.2f\n\n" % test_accuracy)

    percentage_profit = (final_value-initial_value)/initial_value

    f.write("Inicial     : %.2f\n" % initial_value)
    f.write("Final       : %.2f\n" % final_value)
    f.write("Ganancia(%%) : %.2f\n" % percentage_profit)

    net_profit = round(final_value-initial_value,2)
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
