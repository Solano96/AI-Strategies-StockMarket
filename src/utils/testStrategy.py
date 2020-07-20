import pandas as pd
import numpy as np
import math
import datetime

import backtrader as bt
from numpy.random import seed
seed(1)

class TestStrategy(bt.Strategy):
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
        
        # Realimentar red neuronal
        if len(self)-1 >= self.n_day:
            self.model.update_memory(self.X_test[len(self)-self.n_day-1], self.y_test[len(self)-self.n_day-1])
            self.model.reTrain()
            
        # Realizamos la predicción
        p = self.model.predict(self.X_test[len(self)-1])[0][0]
        self.all_predictions.append(p)
        
        # Realizar operacion
        if not self.position:
            if p > 0.6:
                buy_size = self.broker.get_cash() / self.datas[0].open
                self.buy(size = buy_size)

                self.predictions.append(p)
                self.reals.append(np.argmax(self.y_test[len(self)-1])) 
        else:   
            if p < 0.4:
                sell_size = self.broker.getposition(data = self.datas[0]).size
                self.sell(size = sell_size)

                self.predictions.append(p)
                self.reals.append(np.argmax(self.y_test[len(self)-1])) 
        

