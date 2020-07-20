import pandas as pd
import numpy as np
import math
import datetime

import matplotlib
import matplotlib.pyplot as plt

import backtrader as bt
from numpy.random import seed

class TestStrategy(bt.Strategy):
    ''' Esta clase define una estrategia de compra venta '''

    y_test = None
    X_test = None
    model = None
    n_day = None

    predictions = []
    start_to_predict = None

    def log(self, txt, dt=None):
        ''' Método de registro para la estrategia '''
        dt = dt or self.datas[0].datetime.date(0)
        if len(self) > self.start_to_predict:
            print('%s, %s' % (dt.isoformat(), txt))


    def __init__(self):
        ''' Inicializador de la clase TestStrategy '''
        seed(1)
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bought_price = None
        self.day_position = 0

    def notify_order(self, order):
        '''
        Este método es utilizado para notificar las ordenes
        ejecutadas por la estrategia, para ello envía el mensaje de
        notificación correspondiente al método log el cual imprimirá
        la notificación
        '''

        # Orden de compra/venta enviada/aceptada - nada que hacer
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Comprobar si una orden ha sido completada
        # Atención: el broker podría rechazar la orden si no es suficiente dinero
        if order.status in [order.Completed]:
            if order.isbuy(): # Venta
                self.log(
                    'COMPRA EJECUTADA, Precio: %.2f, Coste: %.2f, Comisión: %.2f' %
                    (order.executed.price,
                    order.executed.value,
                    order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else: # Compra
                self.log('VENTA EJECUTADA, Precio: %.2f, Coste: %.2f, Comisión: %.2f' %
                    (order.executed.price,
                    order.executed.value,
                    order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Orden Cancelada/Rechazada')

        # Write down: no pending order
        self.order = None

    def next(self):

        #self.log('Close, %.2f' % self.dataclose[0])

        if len(self) > self.start_to_predict:
            if len(self)-1 >= self.n_day:
                self.model.update_memory(self.X_test[len(self)-self.n_day-1], self.y_test[len(self)-self.n_day-1])
                self.model.reTrain()

        if self.order:
            return

        p = 1

        if len(self) > self.start_to_predict:
            if len(self)-1 < len(self.X_test):
                p = self.model.predict(self.X_test[len(self)-1])[0][0]
                self.predictions.append(p)
        else:
            p = self.predictions[len(self)-1]

        if not self.position:
            if p > 0.6:
                self.log('ORDEN DE COMPRA CREADA, %.2f' % self.dataclose[0])
                buy_size = self.broker.get_cash() / self.datas[0].open
                self.order = self.buy(size = buy_size)
        else:
            if p < 0.4:
                self.log('ORDEN DE VENTA CREADA, %.2f' % self.dataclose[0])
                sell_size = self.broker.getposition(data = self.datas[0]).size
                self.order = self.sell(size = sell_size)

                
