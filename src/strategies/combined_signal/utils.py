'''
Useful functions for combined signal strategy
'''

import numpy as np


def get_split_w_threshold(alpha, normalization='exponential'):
    """
    Get normalize weights and thresholds from alpha vector
    :param alpha: optimize Vectorize
    :return: weights and thresholds
    """
    w = []

    if normalization == 'exponential':
        w = np.exp(alpha[:len(alpha)-2])/np.sum(np.exp(alpha[:len(alpha)-2]))
    elif normalization == 'l1':
        w = alpha[:len(alpha)-2]/np.sum(np.abs(alpha[:len(alpha)-2]))

    buy_threshold = alpha[len(alpha)-2]
    sell_threshold = alpha[len(alpha)-1]

    return w, buy_threshold, sell_threshold


def get_combined_signal(moving_average_rules, moving_averages, w, index):
    """
    Combines in a weighted way buy-sell signals coming from moving average crosses.
    :param moving_average_rules: list with moving average rules
    :param moving_averages: dict with moving averages from historical data
    :param w: weights vector
    :parm index: moving averages index
    :return: final signal get from combined all signals
    """
    signal_list = []

    # Get signals from all moving averages rules
    for short_period, long_period in moving_average_rules:
        moving_average_short = moving_averages['MA_' + str(short_period)][index]
        moving_average_long = moving_averages['MA_' + str(long_period)][index]

        if moving_average_short < moving_average_long:
            signal_list.append(-1)
        else:
            signal_list.append(+1)

    final_signal = np.sum(np.array(w)*np.array(signal_list))

    return final_signal
