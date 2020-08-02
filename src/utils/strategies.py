import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta

import backtrader as bt
from numpy.random import seed
import utils.func_utils as func_utils


class BuyAndHoldStrategy(bt.Strategy):
    """ Buy and Hold Strategy """

    dates = []
    values = []
    closes = []

    def __init__(self):
        """ BuyAndHoldStrategy Class Initializer """
        self.dataclose = self.datas[0].close

    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Buy Operation
        if not self.position:
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)



class ClassicStrategy(bt.SignalStrategy):
    """ Classic Strategy with Moving Averages and RSI """

    params = (
        ('ma_short', 9),
        ('ma_long', 14),
        ('rsi_period', 14),
        ('overbought', 50),
        ('oversold', 50)
    )

    dates = []
    values = []
    closes = []

    def __init__(self):
        """ ClassicStrategy Class Initializer """
        self.dataclose = self.datas[0].close

        ma_short = bt.indicators.SMA(self.datas[0], period=self.params.ma_short)
        ma_long = bt.indicators.SMA(self.datas[0], period=self.params.ma_long)
        self.crossover = ma_short > ma_long
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)


    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Buy Operation
        if not self.position:
            rsi_cross_above_oversold = self.rsi[-1] < self.params.oversold and self.rsi[0] >= self.params.oversold

            if self.crossover and rsi_cross_above_oversold:
                buy_price = self.data.close[0] * (1+0.002)
                buy_size = self.broker.get_cash() / buy_price
                self.buy(size = buy_size)

        # Sell Operation
        else:
            rsi_cross_below_overbought = self.rsi[-1] > self.params.overbought and self.rsi[0] <= self.params.overbought

            if not self.crossover and rsi_cross_below_overbought:
                sell_size = self.broker.getposition(data = self.datas[0]).size
                self.sell(size = sell_size)



class NeuralNetworkStrategy(bt.Strategy):
    """
    This class defines a buying and selling strategy based on the
    prediction of price trends through the use of a neural network.
    """

    y_test = None
    X_test = None
    model = None
    n_day = None

    dates = []
    values = []
    closes = []

    all_predictions = []
    predictions = []
    reals = []

    def __init__(self):
        """ NeuralNetworkStrategy Class Initializer """
        self.dataclose = self.datas[0].close
        self.day_position = 0


    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Predict trend
        p = self.model.predict(self.X_test[len(self)-1])[0][0]
        self.all_predictions.append(p)

        # Buy Operation
        if not self.position and p > 0.55:
            buy_price = self.data.close[0] * (1+0.002)
            buy_size = self.broker.get_cash() / buy_price
            self.buy(size = buy_size)

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))

        # Sell Operation
        elif p < 0.45:
            sell_size = self.broker.getposition(data = self.datas[0]).size
            self.sell(size = sell_size)

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))

        # ReTrain Neural Network
        if len(self) >= self.n_day:
            self.model.update_memory(self.X_test[len(self)-self.n_day], self.y_test[len(self)-self.n_day])
            self.model.reTrain()



class CombinedSignalStrategy(bt.Strategy):
    """
    This class defines a buy-sell strategy based on the combination of moving averages,
    for which it gets a signal by the weighted sum of different signals
    """

    dates = []
    values = []
    closes = []
    w = []
    buy_threshold = None
    sell_threshold = None
    period_list = []
    moving_average_rules = []
    moving_averages = {}

    optimizer = None
    gen_representation = None

    def __init__(self):
        """ CombinedSignalStrategy Class Initializer """
        self.dataclose = self.datas[0].close

    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        '''
        if len(self) % 30 == 0:
            from_date = self.data.datetime.date().replace(year = self.data.datetime.date().year -1)
            to_date = self.data.datetime.date() - timedelta(days=1)

            # Reset best cost
            self.optimizer.swarm.best_cost = 0

            # Optimize weights
            kwargs={'from_date': from_date, 'to_date': to_date}
            best_cost, best_pos = self.optimizer.optimize(self.gen_representation.cost_function, iters=20, **kwargs)

            self.w, self.buy_threshold, self.sell_threshold = func_utils.get_split_w_threshold(best_pos)
        '''

        final_signal = func_utils.get_combined_signal(self.moving_average_rules, self.moving_averages, self.w, len(self)-1)

        # Buy if signal is greater than buy threshold
        if not self.position and final_signal > self.buy_threshold:
            # Get number of shares to buy
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)

        # Sell if singal is smaller than sell threshold
        elif self.position and final_signal < self.sell_threshold:
            sell_size = self.broker.getposition(data = self.datas[0]).size
            self.sell(size = sell_size)
