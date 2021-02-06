# -*- coding: utf-8 -*-
from src.strategies_execution.executions import print_execution_name
from src.strategies_execution.executions import execute_strategy
from src.strategies_execution.executions import optimize_strategy
from src.strategies.moving_averages_cross.strategy_moving_averages_cross import MovingAveragesCrossStrategy


def execute_moving_averages_cross_strategy(df, commission, data_name, start_date, end_date, optimize=False, **kwargs):
    """
    Execute moving averages cross strategy on data history contained in df
    :param df: dataframe with historical data
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :param optimize: if True then optimize strategy
    :return:
        - MAC_Cerebro - execution engine
        - MAC_Strategy - moving averages cross strategy instance
    """

    print_execution_name("Estrategia: cruce de medias móviles")

    strategy_name = 'estrategia_cruce_medias_moviles'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': start_date,
        'Fecha final': end_date
    }

    if optimize:
        print('Optimizando (esto puede tardar)...')

        # Range of values to optimize
        params = {
            'ma_short': range(5, 18),
            'ma_long': range(20, 100, 2)
        }

        # Get best params in past period
        kwargs = optimize_strategy(df, commission, MovingAveragesCrossStrategy, start_date, **params)

    df = df[start_date:end_date]

    MAC_Cerebro = execute_strategy(MovingAveragesCrossStrategy, df, commission, info, **kwargs)

    return MAC_Cerebro, MovingAveragesCrossStrategy
