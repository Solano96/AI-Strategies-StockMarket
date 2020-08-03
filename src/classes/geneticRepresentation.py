import numpy as np
import utils.indicators as indicators
import utils.func_utils as func_utils


class GeneticRepresentation():

    """Genetic representation of a solution and cost function"""

    def __init__(self, df, s_train, e_train, s_test, e_test):
        """ GeneticRepresentation Class Initializer """

        #self.period_list = [2,5,10,15,20,25,30,40,50,75,100,125,150,200,250]
        self.period_list = [5,10,15,20,25,30,40,50]
        self.moving_average_rules = []

        # Get all possible moving averages rules from period list
        for s in self.period_list:
            for l in self.period_list:
                if s < l:
                    self.moving_average_rules.append([s,l])

        self.df = df

        # Add moving average to the DataFrame
        for p in self.period_list:
            self.df = indicators.moving_average(self.df, p)

        # Split DataFrame in train and test
        self.df_train, self.df_test = self.df[s_train:e_train], self.df[s_test:e_test]

        # Drop NaN values
        self.df_train = self.df_train.dropna()
        self.df_test = self.df_test.dropna()

        self.df_closes = [self.df_train.iloc[i]['Close'] for i in range(len(self.df_train.index))]

        self.moving_averages_train = {}
        self.moving_averages_test = {}

        # Vectorize columns and save in dict to fast access
        for p in self.period_list:
            col_ma_name = 'MA_' + str(p)
            self.moving_averages_train[col_ma_name] = [self.df_train.iloc[i][col_ma_name] for i in range(len(self.df_train.index))]
            self.moving_averages_test[col_ma_name] = [self.df_test.iloc[i][col_ma_name] for i in range(len(self.df_test.index))]


    def cost_function(self, x, from_date, to_date, normalization='exponential'):
        """ Cost function adapted to PSO algorithm """

        df_evaluate = self.df[from_date:to_date]
        df_closes = [df_evaluate.iloc[i]['Close'] for i in range(len(df_evaluate.index))]

        moving_averages_evaluate = {}

        # Vectorize columns and save in dict to fast access
        for p in self.period_list:
            col_ma_name = 'MA_' + str(p)
            moving_averages_evaluate[col_ma_name] = [df_evaluate.iloc[i][col_ma_name] for i in range(len(df_evaluate.index))]

        # Get the number of particles of PSO
        num_particles = x.shape[0]
        final_prices = np.zeros([num_particles])

        for idx, alpha in enumerate(x):
            size = len(df_closes)
            in_market = False

            buy_day_list = []
            sell_day_list = []

            w, buy_threshold, sell_threshold = func_utils.get_split_w_threshold(alpha, normalization)

            for i in range(size):

                if in_market and i == size-1:
                    sell_day_list.append(i)

                elif i < size-1:

                    final_signal = func_utils.get_combined_signal(self.moving_average_rules, moving_averages_evaluate, w, i)

                    if final_signal > buy_threshold and not in_market:
                        in_market = True
                        buy_day_list.append(i)

                    elif final_signal < sell_threshold and in_market:
                        in_market = False
                        sell_day_list.append(i)

            num_trades = len(buy_day_list)
            commission = 0.001
            start_price = 100000
            final_price = start_price

            # Get the final capital after excute all trades
            for i in range(num_trades):
                final_price *= (df_closes[sell_day_list[i]]*(1-commission)) / (df_closes[buy_day_list[i]]*(1+commission))

            final_prices[idx] = final_price

        return -final_prices
