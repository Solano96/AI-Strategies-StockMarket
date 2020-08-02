import backtrader as bt


class BuyAndHoldStrategy(bt.Strategy):
    """ Buy and Hold Strategy """

    dates = []
    values = []
    closes = []

    def __init__(self):
        """ BuyAndHoldStrategy Class Initializer """
        self.dataclose = self.datas[0].close

    def next(self):
        """ Define logic in each iteration """

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Buy Operation
        if not self.position:
            buy_size = self.broker.get_cash() / self.datas[0].open
            self.buy(size = buy_size)
