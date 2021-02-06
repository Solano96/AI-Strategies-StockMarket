import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta

import backtrader as bt
from numpy.random import seed
from src.strategies.combined_signal.utils import *
from src.strategies.log_strategy import LogStrategy


class CombinedSignalStrategy(LogStrategy):
    """
    This class defines a buy-sell strategy based on the combination of moving averages,
    for which it gets a signal by the weighted sum of different signalsself.

    To use this strategy is neccesary to provided some parameters:

    :param w: weights vector
    :type w: list of float

    :param buy_threshold: threshold used to determine whether a purchase order is submitted
    :type buy_threshold: float

    :param sell_threshold: threshold used to determine whether a sales order is sent
    :type sell_threshold: float

    :param moving_average_rules: list with moving average rules
    :type moving_average_rules: list of int

    :param moving_averages: dict with moving averages from historical data where index is the period of ma
    :type moving_averages: dict of (int, list of float)

    :param retrain_params: dict with retraining configuration
    :type retrain_params: dict of int
    """

    # Parameters required for the strategy
    w = []
    buy_threshold = None
    sell_threshold = None
    moving_average_rules = []
    moving_averages = {}
    normalization = None

    optimizer = None
    gen_representation = None

    retrain_params = {
        'repeat': 90,
        'interval': 100,
        'iters': 10
    }

    dates = []
    values = []
    closes = []


    def __init__(self):
        """ CombinedSignalStrategy Class Initializer """
        super().__init__()


    def next(self):
        """ Define logic in each iteration """
        self.log_close_price()

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # we cannot send a 2nd an order if if another one is pending
        if self.order:
            return

        # ReTrain optimization algorithm
        if self.optimizer != None:

            retrain_optimizer = (len(self) % self.retrain_params['repeat'] == 0)

            if retrain_optimizer:
                # Get training dates
                days_ago = self.retrain_params['interval']
                from_date = self.data.datetime.date() - timedelta(days=days_ago)
                to_date = self.data.datetime.date() - timedelta(days=1)

                # Reset best cost
                self.optimizer.swarm.best_cost = 0

                # Optimize weights
                iters = self.retrain_params['iters']
                kwargs={'from_date': from_date, 'to_date': to_date}
                best_cost, best_pos = self.optimizer.optimize(self.gen_representation.cost_function, iters=iters, **kwargs)

                self.w, self.buy_threshold, self.sell_threshold = get_split_w_threshold(best_pos, self.normalization)

        # Get combined signal
        final_signal = get_combined_signal(self.moving_average_rules, self.moving_averages, self.w, len(self)-1)

        # Buy if signal is greater than buy threshold
        if not self.position and final_signal > self.buy_threshold:
            self.buy()

        # Sell if singal is smaller than sell threshold
        elif self.position and final_signal < self.sell_threshold:
            self.sell()
