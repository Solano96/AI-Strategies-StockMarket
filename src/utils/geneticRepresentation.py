import numpy as np
import utils.indicators as indicators


class GeneticRepresentation():

    """Genetic representation of a solution and cost function"""

    def __init__(self, df, s_train, e_train, s_test, e_test):
        self.period_list = [2,5,10,15,20,25,30,40,50,75,100,125,150,200,250]
        self.moving_average_rules = []

        for s in self.period_list:
            for l in self.period_list:
                if s < l:
                    self.moving_average_rules.append([s,l])

        self.df = df

        for p in self.period_list:
            self.df = indicators.moving_average(self.df, p)


        self.df_train, self.df_test = self.df[s_train:e_train], self.df[s_test:e_test]

        self.df_train = self.df_train.dropna()
        self.df_test = self.df_test.dropna()

        self.df_closes = [self.df_train.iloc[i]['Close'] for i in range(len(self.df_train.index))]

        self.moving_averages_train = {}
        self.moving_averages_test = {}

        for p in self.period_list:
            col_ma_name = 'MA_' + str(p)
            self.moving_averages_train[col_ma_name] = [self.df_train.iloc[i][col_ma_name] for i in range(len(self.df_train.index))]
            self.moving_averages_test[col_ma_name] = [self.df_test.iloc[i][col_ma_name] for i in range(len(self.df_test.index))]


    def cost_function(self, x):

        num_particles = x.shape[0]
        final_prices = np.zeros([num_particles])

        for idx, alpha in enumerate(x):
            size = len(self.df_closes)
            in_market = False

            buy_day_list = []
            sell_day_list = []

            #w = np.exp(alpha)/np.sum(np.exp(alpha))
            w = alpha/np.sum(alpha)
            #w = alpha

            for i in range(size):

                if in_market and i == size-1:
                    sell_day_list.append(i)

                elif i < size-1:
                    signal_list = []

                    for short_period, long_period in self.moving_average_rules:
                        moving_average_short = self.moving_averages_train['MA_' + str(short_period)][i]
                        moving_average_long = self.moving_averages_train['MA_' + str(long_period)][i]

                        if moving_average_short < moving_average_long:
                            signal_list.append(-1)
                        else:
                            signal_list.append(+1)

                    final_signal = 0

                    for w_i, s_i in zip(w, signal_list):
                        final_signal += w_i*s_i

                    if final_signal > 0.2 and not in_market:
                        in_market = True
                        buy_day_list.append(i)

                    elif final_signal < -0.2 and in_market:
                        in_market = False
                        sell_day_list.append(i)

            num_trades = len(buy_day_list)
            commission = 0.001
            start_price = 100000
            final_price = start_price

            for i in range(num_trades):
                final_price *= (self.df_closes[sell_day_list[i]]*(1-commission)) / (self.df_closes[buy_day_list[i]]*(1+commission))

            final_prices[idx] = final_price

        return -final_prices
