import backtrader as bt
from src.strategies.log_strategy import LogStrategy


class BuyAndHoldStrategy(LogStrategy):
    """ Buy and Hold Strategy """


    def __init__(self):
        """ BuyAndHoldStrategy Class Initializer """
        super().__init__()


    def next(self):
        """ Define logic in each iteration """
        self.update_log_values()

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Buy Operation
        if not self.position:
            self.send_buy_order()
