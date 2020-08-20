from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from backtrader import Analyzer
from backtrader.utils import AutoOrderedDict, AutoDict
from backtrader.utils.py3 import MAXINT


class MyAnalyzer(Analyzer):


    def create_analysis(self):
        self.rets = AutoOrderedDict()
        self.rets.trades.total = 0
        self.rets.trades.positives = 0
        self.rets.trades.negatives = 0

        self.accumulate = 0
        self.accumulate_profit = 0
        self.accumulate_loss = 0


        self.rets.avg.trade = 0
        self.rets.avg.profit_trade = 0
        self.rets.avg.loss_trade = 0

        self._value = self.strategy.broker.get_cash()

    def stop(self):
        super(MyAnalyzer, self).stop()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        trade_return = (self.strategy.broker.get_cash()-self._value)/self._value

        self.rets.trades.total+=1
        self.accumulate += trade_return
        self.rets.avg.trade = self.accumulate/self.rets.trades.total

        if trade.pnlcomm > 0:
            self.rets.trades.positives+=1
            self.accumulate_profit += trade_return
            self.rets.avg.profit_trade = self.accumulate_profit/self.rets.trades.positives
        elif trade.pnlcomm < 0:
            self.rets.trades.negatives+=1
            self.accumulate_loss += trade_return
            self.rets.avg.loss_trade = self.accumulate_loss/self.rets.trades.negatives

        self._value = self.strategy.broker.get_cash()
