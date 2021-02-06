# -*- coding: utf-8 -*-
from src.strategies_execution.executions import print_execution_name
from src.strategies_execution.executions import execute_strategy
from src.strategies_execution.executions import optimize_strategy
from src.strategies.one_moving_average.strategy_one_moving_average import OneMovingAverageStrategy


def execute_one_moving_average_strategy(df, commission, data_name, start_date, end_date, optimize=False, **kwargs):
    """
    Execute one moving average strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - OMA_Cerebro - execution engine
        - OMA_Strategy - one moving average strategy instance
    """

    print_execution_name("Estrategia: media móvil")

    strategy_name = 'estrategia_media_movil'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    if optimize:
        print('Optimizando (esto puede tardar)...')

        params = {'ma_period': range(5, 50)}

        # Get best params in past period
        kwargs = optimize_strategy(df, commission, OneMovingAverageStrategy, start_date, **params)

    df = df[start_date:end_date]

    OMA_Strategy =  OneMovingAverageStrategy
    OMA_Cerebro = execute_strategy(OMA_Strategy, df, commission, info, **kwargs)

    return OMA_Cerebro, OMA_Strategy
