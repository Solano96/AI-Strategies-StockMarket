import backtrader as bt
from src.strategies.log_strategy import LogStrategy


class MovingAveragesCrossStrategy(LogStrategy):
    """ Moving averages cross Strategy """

    params = (
        ('ma_short', 5),
        ('ma_long', 20),
    )

    dates = []
    values = []
    closes = []


    def __init__(self):
        """ MovingAveragesCrossStrategy Class Initializer """
        super().__init__()

        # Simple Moving Average Indicator short and long period
        ma_short = bt.indicators.SMA(self.datas[0], period=self.params.ma_short)
        ma_long = bt.indicators.SMA(self.datas[0], period=self.params.ma_long)
        # Crossover signal
        self.crossover = ma_short > ma_long


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
            if self.crossover:
                self.buy()
        # If we are not in the market
        else:
            if not self.crossover:
                self.sell()


    def stop(self):
        self.log('(MA short Period %2d, MA long Period %2d) Ending Value %.2f' %
                 (self.params.ma_short, self.params.ma_long, self.broker.getvalue()))
