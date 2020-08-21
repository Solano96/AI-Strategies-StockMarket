import backtrader as bt
from src.strategies.log_strategy import LogStrategy

class OneMovingAverageStrategy(LogStrategy):
    """ One Moving Average Strategy """

    params = (
        ('maperiod', 15),
    )


    def __init__(self):
        """ OneMovingAverageStrategy Class Initializer """
        super().__init__()

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)


    def next(self):
        """ Define logic in each iteration """        
        self.update_log_values()

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                self.send_buy_order()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.send_sell_order()


    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)
