# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import utils function
import src.utils.func_utils as func_utils

# Import classes
from src.classes.myCerebro import MyCerebro
from src.classes.myAnalyzer import MyAnalyzer
from src.classes.myBuySell import MyBuySell
from src.classes.maxRiskSizer import MaxRiskSizer

# Import strategies execution
import src.strategies_execution.execution_analysis as execution_analysis
import src.strategies_execution.execution_plot as execution_plot

# Import strategy
from src.strategies_execution.executions import print_execution_name
from src.strategies_execution.executions import execute_strategy
from src.strategies.buy_and_hold.strategy_buy_and_hold import BuyAndHoldStrategy

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

    return BH_Cerebro, BH_Strategy
