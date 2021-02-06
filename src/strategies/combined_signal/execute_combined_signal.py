# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime, timedelta
import pyswarms as ps

# Import utils function
from src.strategies.combined_signal.utils import *

# Import strategy
from src.strategies_execution.executions import print_execution_name
from src.strategies_execution.executions import execute_strategy
from src.strategies.combined_signal.genetic_representation import GeneticRepresentation
from src.strategies.combined_signal.strategy_combined_signal import CombinedSignalStrategy


def execute_pso_strategy(df, options, topology, retrain_params, commission, data_name, s_test, e_test, iters=100, normalization='exponential'):
    """
    Execute particle swarm optimization strategy on data history contained in df
    :param df: dataframe with historical data
    :param options: dict with the following parameters
        - c1 - cognitive parameter with which the particle follows its personal best
        - c2 - social parameter with which the particle follows the swarm's global best position
        - w - parameter that controls the inertia of the swarm's movement
    :param commision: commission to be paid on each operation
    :param data_name: quote data name
    :param start_date: start date of simulation
    :param end_date: end date of simulation
    :return:
        - PSO_Cerebro - execution engine
        - PSO_Strategy - pso strategy instance
    """

    print_execution_name("Estrategia: particle swar optimization")

    strategy_name = 'particle_swarm_optimization'

    info = {
        'Mercado': data_name,
        'Estrategia': strategy_name,
        'Fecha inicial': s_test,
        'Fecha final': e_test
    }

    # ------------ Obtenemos los conjuntos de train y test ------------ #

    s_test_date = datetime.strptime(s_test, '%Y-%m-%d')
    s_train = s_test_date.replace(year = s_test_date.year - 2)
    #s_train = s_test_date - timedelta(days=180)
    e_train = s_test_date - timedelta(days=1)

    gen_representation = GeneticRepresentation(df, s_train, e_train, s_test, e_test)

    # ------------ Fijamos hiperparámetros ------------ #

    n_particles = topology['particles']
    num_neighbours = topology['neighbours']
    minkowski_p_norm = 2
    options['k'] = num_neighbours
    options['p'] = minkowski_p_norm
    dimensions=len(gen_representation.moving_average_rules)+2

    if normalization == 'exponential':
        max_bound = 2.0 * np.ones(dimensions-2)
        min_bound = -max_bound
    elif normalization == 'l1':
        max_bound = 2.0 * np.ones(dimensions-2)
        min_bound = np.zeros(dimensions-2)

    max_bound = np.append(max_bound, [0.9, 0.0])
    min_bound = np.append(min_bound, [0.0, -0.9])
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=n_particles,
                                        dimensions=dimensions,
                                        options=options,
                                        bounds=bounds,
                                        static=True)

    # Perform optimization
    kwargs={'from_date': s_train, 'to_date': e_train, 'normalization': normalization}
    best_cost, best_pos = optimizer.optimize(gen_representation.cost_function,
                                             iters=iters,
                                             n_processes=2,
                                             **kwargs)

    # Create an instance from CombinedSignalStrategy class and assign parameters
    PSO_Strategy = CombinedSignalStrategy
    w, buy_threshold, sell_threshold = get_split_w_threshold(best_pos)

    """
    print("Umbral de compra: ", buy_threshold)
    print("Umbral de venta: ", sell_threshold)

    crosses = ["(" + str(cross[0]) + ", " + str(cross[1]) + ")" for cross in gen_representation.moving_average_rules]

    y_pos = np.arange(len(crosses))
    plt.bar(y_pos, w)
    plt.xticks(y_pos, crosses)
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(top=0.98, bottom=0.2, left=0.08, right=0.98, hspace=0.0, wspace=0.0)
    plt.show()
    """

    PSO_Strategy.w = w
    PSO_Strategy.buy_threshold = buy_threshold
    PSO_Strategy.sell_threshold = sell_threshold
    PSO_Strategy.moving_average_rules = gen_representation.moving_average_rules
    PSO_Strategy.moving_averages = gen_representation.moving_averages_test
    PSO_Strategy.optimizer = optimizer
    PSO_Strategy.gen_representation = gen_representation
    PSO_Strategy.normalization = normalization
    PSO_Strategy.retrain_params = retrain_params

    df_test = gen_representation.df_test
    df_train = gen_representation.df_train

    PSO_Cerebro = execute_strategy(PSO_Strategy, df_test, commission, info, retrain_params)

    return PSO_Cerebro, PSO_Strategy
