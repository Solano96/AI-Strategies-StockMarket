# -*- coding: utf-8 -*-

import os
import sys

import backtrader.plot
import matplotlib
import matplotlib.pyplot as plt

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def create_folder_inside_img_if_not_exists(folder_name):
    """
    Create folder inside img folder
    :param folder_name: folder name
    """
    if not os.path.exists('../img'):
        os.makedirs('../img')

    if not os.path.exists('../img/' + folder_name):
        os.makedirs('../img/' + folder_name)


def plot_simulation(cerebro, file_name, data_name, from_date=None, to_date=None, size=None, style='line'):
    """
    Plot strategy simulation
    :param cerebro: backtrader engine
    :param file_name: file name for the generated image
    :param data_name: quote data name
    :param from_date: start date of simulation
    :param to_date: end date of simulation
    :return: saved file name
    """
    cerebro.getFig(iplot=False, style=style, barup='green')

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)

    fig = matplotlib.pyplot.gcf()

    default_size_inches = fig.get_size_inches()

    if size != None:
        fig.set_size_inches(size)
    else:
        size = default_size_inches

    #plt.show()

    # Create simulacion folder if not exists
    create_folder_inside_img_if_not_exists('simulacion_' + file_name)

    saved_file_name = '../img/simulacion_' + file_name + '/' + data_name + '_'

    if from_date != None and to_date != None:
        saved_file_name += from_date + '_' + to_date + '_'

    saved_file_name += file_name + '_' + str(size[0]) + '_' + str(size[1]) + '.png'

    plt.savefig(saved_file_name)
    fig.set_size_inches(default_size_inches)

    return saved_file_name


def plot_capital(strategy_list, data_name, img_name, from_date=None, to_date=None):
    """
    Plot chart with the capital of the strategy list
    :param strategy_list: list with the strategies and their respective names
    :param data_name: quote data name
    :param img_name: file name for the generated image
    :param from_date: start date of simulation
    :param to_date: end date of simulation
    """
    fig = plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.9, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(111)

    for strategy, name_strategy in strategy_list:
        ax.plot(strategy.dates, strategy.values, label=name_strategy)

    ax.legend(loc='upper left')
    ax.yaxis.grid(linestyle="-")

    # Create ganancias folder if not exists
    create_folder_inside_img_if_not_exists('ganancias')

    saved_file_name = '../img/ganancias/' + data_name + '_'

    if from_date != None and to_date != None:
        saved_file_name += from_date + '_' + to_date + '_'

    saved_file_name += img_name + '.png'

    plt.savefig(saved_file_name)
