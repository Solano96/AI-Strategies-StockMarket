# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys, getopt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import src.utils.func_utils as func_utils

# Import classes
from src.classes.myCerebro import MyCerebro
from src.classes.myAnalyzer import MyAnalyzer
from src.classes.myBuySell import MyBuySell
from src.classes.maxRiskSizer import MaxRiskSizer

# Import strategies execution
import src.strategies_execution.execution_analysis as execution_analysis
import src.strategies_execution.execution_plot as execution_plot

import backtrader as bt
import backtrader.plot


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
    # bt.observers.BuySell = MyBuySell

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
    execution_plot.plot_simulation(cerebro, info)

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

    best_params = dict()
    best_value = 0

    # Search best parameters
    for stratrun in stratruns: # Iter executions
        for strat in stratrun: # Iter Strategy in execution
            # Get final cash
            final_value = strat.analyzers[0]._value_end

            #print(final_value, ": ", strat.p._getkwargs())

            # Update best params
            if best_value < final_value:
                best_value = final_value
                best_params = strat.p._getkwargs()

    return best_params
