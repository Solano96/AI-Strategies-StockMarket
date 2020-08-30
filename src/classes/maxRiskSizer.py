import backtrader as bt
import math
import logging


class MaxRiskSizer(bt.Sizer):
    '''
    Returns the number of shares rounded down that can be purchased for the
    max risk tolerance
    '''

    params = (('risk', 0.1),
                ('debug', True))


    def log(self, txt, data):
        ''' Logging function fot this strategy'''
        dt = data.datetime.date(0)
        logging.info('Date: %s, %s' % (dt.isoformat(), txt))


    def _getsizing(self, comminfo, cash, data, isbuy):
        '''
        Method to get order size.
        :param comminfo: The CommissionInfo instance that contains information
          about the commission for the data and allows calculation of position
          value, operation cost, commision for the operation
        :param cash: current available cash in the *broker*
        :param data: target of the operation
        :param isbuy: will be `True` for *buy* operations and `False` for *sell* operations
        :return: actual size (an int) to be executed.
        '''

        size = 0
        # Work out the maximum size assuming all cash can be used.
        max_risk = cash * self.p.risk
        # Get commission
        comm = comminfo.p.commission
        # Apply the commission to the price. We can then divide our risk
        # by this value
        com_adj_price = data[0] * (1 + (comm * 2)) # *2 for round trip

        if isbuy:
            comm_adj_size = max_risk / com_adj_price
            #Avoid accidentally going short
            if comm_adj_size < 0:
                comm_adj_size = 0

            #Finally make sure we round down to the nearest unit.
            comm_adj_size = math.floor(comm_adj_size)
        else:
            # sell all shares
            comm_adj_size = self.broker.getposition(data).size

        if self.p.debug:
            if isbuy:
                buysell = 'Buying'
                self.log('BUY CREATE, %.2f' % data[0], data)
            else:
                buysell = 'Selling'
                self.log('SELL CREATE, %.2f' % data[0], data)

            logging.info("------------- SIZER INFO --------------")
            logging.info("Action: {}".format(buysell))
            logging.info("Price: {}".format(data[0]))
            logging.info("Cash: {0:.2f}".format(cash))
            logging.info("Max Risk %: {}".format(self.p.risk*100))
            logging.info("Max Risk $: {0:.2f}".format(max_risk))
            logging.info("Commission: {}".format(comm))
            logging.info("Commission Adj Price (Round Trip): {0:.2f}".format(com_adj_price))
            logging.info("Size: {}".format(comm_adj_size))
            logging.info("----------------------------------------")

        return comm_adj_size
