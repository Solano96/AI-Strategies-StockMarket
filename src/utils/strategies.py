import pandas as pd
import numpy as np
import math
import datetime

import backtrader as bt
from numpy.random import seed


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



class ClassicStrategy(bt.Strategy):
    """ Classic Strategy with Moving Averages and RSI """

    dates = []
    values = []
    closes = []

    def __init__(self):
        """ ClassicStrategy Class Initializer """
        self.dataclose = self.datas[0].close

        self.ema_9 = bt.indicators.EMA(self.datas[0], period=9)
        self.ema_14 = bt.indicators.EMA(self.datas[0], period=14)
        self.rsi_14 = bt.indicators.RSI_EMA(self.datas[0], period = 14)

    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Buy Operation
        if not self.position:
            if self.ema_9[0] > self.ema_14[0] and self.rsi_14[-1] > 70 and self.rsi_14[0] <= 70:
                buy_size = self.broker.get_cash() / self.datas[0].open
                self.buy(size = buy_size)

        # Sell Operation
        else:
            if self.ema_9[0] < self.ema_14[0] and self.rsi_14[-1] < 30 and self.rsi_14[0] >= 30:
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
            buy_size = self.broker.get_cash() / self.datas[0].open
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
    w = None
    period_list = []
    moving_average_rules = []
    moving_averages = {}

    def __init__(self):
        """ CombinedSignalStrategy Class Initializer """
        self.dataclose = self.datas[0].close

    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        signal_list = []

        # Get signals from all moving averages rules
        for short_period, long_period in self.moving_average_rules:
            moving_average_short = self.moving_averages['MA_' + str(short_period)][len(self)-1]
            moving_average_long = self.moving_averages['MA_' + str(long_period)][len(self)-1]

            if moving_average_short < moving_average_long:
                signal_list.append(-1)
            else:
                signal_list.append(+1)

        final_signal = 0

        # Get a unique signal from the weighted sum of all signals
        for w_i, s_i in zip(self.w, signal_list):
            final_signal += w_i*s_i

        # Buy if signal is greater than 0.2
        if not self.position and final_signal > 0.2:
            # Get number of shares to buy
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)

        # Sell if singal is smaller than -0.2
        elif self.position and final_signal < -0.2:
            sell_size = self.broker.getposition(data = self.datas[0]).size
            self.sell(size = sell_size)
