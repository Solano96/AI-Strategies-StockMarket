from src.strategies.log_strategy import LogStrategy


class BuyAndHoldStrategy(LogStrategy):
    """ Buy and Hold Strategy """

    dates = []
    values = []
    closes = []

    def __init__(self):
        """ BuyAndHoldStrategy Class Initializer """
        super().__init__()


    def next(self):
        """ Define logic in each iteration """
        self.log_close_price()

        self.values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date())
        self.closes.append(self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Buy Operation
        if not self.position:
            self.buy()
