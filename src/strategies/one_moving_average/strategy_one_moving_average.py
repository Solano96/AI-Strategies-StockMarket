import backtrader as bt
from src.strategies.log_strategy import LogStrategy

class OneMovingAverageStrategy(LogStrategy):
    """ One Moving Average Strategy """

    params = (
        ('ma_period', 15),
    )

    dates = []
    values = []
    closes = []


    def __init__(self):
        """ OneMovingAverageStrategy Class Initializer """
        super().__init__()

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period)


    def next(self):
        """ Define logic in each iteration """
        self.log_close_price()

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.sell()


    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.ma_period, self.broker.getvalue()))
