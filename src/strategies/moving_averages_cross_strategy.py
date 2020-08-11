import backtrader as bt
from strategies.log_strategy import LogStrategy

class MovingAveragesCrossStrategy(LogStrategy):
    """ Moving averages cross Strategy """

    dates = []
    values = []
    closes = []

    params = (
        ('ma_short', 5),
        ('ma_long', 20),
        ('printlog', False),
    )


    def __init__(self):
        """ MovingAveragesCrossStrategy Class Initializer """
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Simple Moving Average Indicator short and long period
        ma_short = bt.indicators.SMA(self.datas[0], period=self.params.ma_short)
        ma_long = bt.indicators.SMA(self.datas[0], period=self.params.ma_long)
        # Crossover signal
        self.crossover = ma_short > ma_long


    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

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
                # Log buy order
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                buy_price = self.data.close[0] * (1+0.002)
                buy_size = self.broker.get_cash() / buy_price

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(size = buy_size)

        else:

            if not self.crossover:
                # Log sell order
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                sell_size = self.broker.getposition(data = self.datas[0]).size

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size = sell_size)


    def stop(self):
        self.log('(MA short Period %2d, MA long Period %2d) Ending Value %.2f' %
                 (self.params.ma_short, self.params.ma_long, self.broker.getvalue()), doprint=True)
