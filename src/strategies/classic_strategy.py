import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta

import backtrader as bt
from numpy.random import seed
import src.utils.func_utils as func_utils


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
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders
        self.order = None

        # Simple Moving Average Indicator short and long period
        ma_short = bt.indicators.SMA(self.datas[0], period=self.params.ma_short)
        ma_long = bt.indicators.SMA(self.datas[0], period=self.params.ma_long)
        # Crossover signal
        self.crossover = ma_short > ma_long
        # RSI Indicator
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)


    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # we cannot send a 2nd an order if if another one is pending
        if self.order:
            return

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
