import pandas as pd
import numpy as np
import math
import datetime

import backtrader as bt
from numpy.random import seed


class BuyAndHoldStrategy(bt.Strategy):
    ''' Esta clase define una estrategia de compra venta '''

    dates = []
    values = []
    closes = []

    def __init__(self):
        ''' Inicializador de la clase TestStrategy '''
        self.dataclose = self.datas[0].close

    def next(self):

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Realizar operacion
        if not self.position:
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)



class ClassicStrategy(bt.Strategy):
    ''' Esta clase define una estrategia de compra venta '''

    dates = []
    values = []
    closes = []

    def __init__(self):
        ''' Inicializador de la clase TestStrategy '''
        self.dataclose = self.datas[0].close

        self.ema_9 = bt.indicators.EMA(self.datas[0], period=9)
        self.ema_14 = bt.indicators.EMA(self.datas[0], period=14)
        self.rsi_14 = bt.indicators.RSI_EMA(self.datas[0], period = 14)

    def next(self):

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Realizar operacion
        if not self.position:
            if self.ema_9[0] > self.ema_14[0] and self.rsi_14[-1] > 70 and self.rsi_14[0] <= 70:
                buy_size = self.broker.get_cash() / self.datas[0].open
                self.buy(size = buy_size)
        else:
            if self.ema_9[0] < self.ema_14[0] and self.rsi_14[-1] < 30 and self.rsi_14[0] >= 30:
                sell_size = self.broker.getposition(data = self.datas[0]).size
                self.sell(size = sell_size)



class NeuralNetworkStrategy(bt.Strategy):
    ''' Esta clase define una estrategia de compra venta '''

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
        ''' Inicializador de la clase TestStrategy '''
        self.dataclose = self.datas[0].close
        self.day_position = 0

    def next(self):

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Realizamos la predicción
        p = self.model.predict(self.X_test[len(self)-1])[0][0]
        self.all_predictions.append(p)

        # Realizar operacion
        if not self.position and p > 0.55:
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))
        elif p < 0.45:
            sell_size = self.broker.getposition(data = self.datas[0]).size
            self.sell(size = sell_size)

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))

        # Realimentar red neuronal
        if len(self) >= self.n_day:
            self.model.update_memory(self.X_test[len(self)-self.n_day], self.y_test[len(self)-self.n_day])
            self.model.reTrain()
