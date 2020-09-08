import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta

import backtrader as bt
from numpy.random import seed
import src.utils.func_utils as func_utils
from src.strategies.log_strategy import LogStrategy


class NeuralNetworkStrategy(LogStrategy):
    """
    This class defines a buying and selling strategy based on the
    prediction of price trends through the use of a neural network.
    """

    y_test = None
    X_test = None
    model = None
    n_day = None

    all_predictions = []
    predictions = []
    reals = []

    dates = []
    values = []
    closes = []


    def __init__(self):
        """ NeuralNetworkStrategy Class Initializer """
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

        # Predict trend
        p = self.model.predict(self.X_test[len(self)-1])[0][0]
        self.all_predictions.append(p)

        # Buy Operation
        if not self.position and p > 0.55:
            self.buy()

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))

        # Sell Operation
        elif p < 0.45:
            self.sell()

            self.predictions.append(p)
            self.reals.append(np.argmax(self.y_test[len(self)-1]))

        # ReTrain Neural Network
        if len(self) >= self.n_day:
            self.model.update_memory(self.X_test[len(self)-self.n_day], self.y_test[len(self)-self.n_day])
            self.model.reTrain()
