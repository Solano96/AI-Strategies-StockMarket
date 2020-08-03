# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys

import backtrader as bt

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import strategies_execution.execution_plot as execution_plot

from fpdf import FPDF


def create_folder_if_not_exists(folder_name):
    """
    Create folder inside img folder
    :param folder_name: folder name
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def printAnalysis(file_name, data_name, initial_value, final_value, tradeAnalyzer, drawDownAnalyzer, myAnalyzer,
                  train_accuracy=None, test_accuracy=None):
    '''
    Function to print the Technical Analysis results in a nice format.
    :param file_name: file name to print the analysis
    :param data_name: quote data name
    :param initial_value: initial value of the portfolio
    :param final_value: final value of the portfolio
    :param tradeAnalyzer: trade analyzer instance
    :param drawDownAnalyzer: drawdown analyzer instance
    :param myAnalyzer: myAnalyzer instance
    :param train_accuracy: train accuracy (optional)
    :param test_accuracy: test accuracy (optional)
    '''

    create_folder_if_not_exists('../resultados')

    f = open ('../resultados/resultados_' + file_name + '.txt','a')
    f.write(data_name)
    f.write("\n\n")

    if train_accuracy != None and test_accuracy != None:
        f.write("Train score : %.2f\n" % train_accuracy)
        f.write("Test score  : %.2f\n\n" % test_accuracy)

    percentage_profit = (final_value-initial_value)/initial_value

    f.write("Inicial     : %.2f\n" % initial_value)
    f.write("Final       : %.2f\n" % final_value)
    f.write("Ganancia(%%) : %.2f\n" % percentage_profit)

    net_profit = round(final_value-initial_value,2)
    maxdd = round((-1.0)*drawDownAnalyzer.max.drawdown,2)
    trades_total = int(myAnalyzer.trades.total)
    trades_positives = int(myAnalyzer.trades.positives)
    trades_negatives = int(myAnalyzer.trades.negatives)
    avg_trade = round(myAnalyzer.avg.trade,2)
    avg_profit_trade = round(myAnalyzer.avg.profit_trade,2)
    avg_loss_trade = round(myAnalyzer.avg.loss_trade,2)

    avg_profit_loss = 99999999

    if avg_loss_trade != 0:
        avg_profit_loss = round((-1)*avg_profit_trade/avg_loss_trade,2)

    f.write("Ganancias   : %.2f\n" % net_profit)
    f.write("Max DD      : %.2f\n" % maxdd)
    f.write("Trades total: %i\n" % trades_total)
    f.write("Trades+     : %i\n" % trades_positives)
    f.write("Trades-     : %i\n" % trades_negatives)
    f.write("Avg trade   : %.2f\n" % avg_trade)
    f.write("Avg profit  : %.2f\n" % avg_profit_trade)
    f.write("Avg loss    : %.2f\n" % avg_loss_trade)
    f.write("Profit/Loss : %.2f\n\n\n" % avg_profit_loss)

    f.close()


def print_section(pdf, text_section, font_family, section_size, margin):
    pdf.ln(0.5)
    #pdf.set_text_color(0,75,126)
    pdf.set_font(font_family, style='B', size=section_size)
    pdf.cell(margin, 0, txt=text_section)
    pdf.set_font(font_family, style='', size=12)
    pdf.set_text_color(0,0,0)
    pdf.ln(0.4)


def printAnalysisPDF(cerebro, file_name, data_name, initial_value, final_value, tradeAnalyzer, drawDownAnalyzer, myAnalyzer,
                     from_date=None, to_date=None):
    '''
    Function to generate a report in PDF format.
    :param cerebro: backtrader engine
    :param file_name: file name to print the analysis
    :param data_name: quote data name
    :param initial_value: initial value of the portfolio
    :param final_value: final value of the portfolio
    :param tradeAnalyzer: trade analyzer instance
    :param drawDownAnalyzer: drawdown analyzer instance
    :param myAnalyzer: myAnalyzer instance
    :param from_date: start date of simulation
    :param to_date: end date of simulation
    '''
    percentage_profit = (final_value-initial_value)/initial_value

    net_profit = round(final_value-initial_value,2)
    maxdd = round((-1.0)*drawDownAnalyzer.max.drawdown,2)
    trades_total = int(myAnalyzer.trades.total)
    trades_positives = int(myAnalyzer.trades.positives)
    trades_negatives = int(myAnalyzer.trades.negatives)
    avg_trade = round(myAnalyzer.avg.trade,2)
    avg_profit_trade = round(myAnalyzer.avg.profit_trade,2)
    avg_loss_trade = round(myAnalyzer.avg.loss_trade,2)

    avg_profit_loss = 99999999

    if avg_loss_trade != 0:
        avg_profit_loss = round((-1)*avg_profit_trade/avg_loss_trade,2)

    pdf = FPDF(unit='in')
    effective_page_width = pdf.w - 2*pdf.l_margin
    sep = effective_page_width/4.0
    section_size = 14
    font_family = 'Helvetica'

    pdf.add_page()
    pdf.set_font(font_family, style='B', size=28)

    pdf.set_line_width(0.02)
    #pdf.set_draw_color(0,75,126)
    #pdf.set_draw_color(200,10,10)
    pdf.line(0.4, 0.5, 7.8, 0.5)
    pdf.line(0.4, 1.4, 7.8, 1.4)
    pdf.line(0.4, 11.2, 7.8, 11.2)

    margin = sep
    line_sep = 0.3
    pdf.ln(0.6)
    pdf.cell(margin, 0, txt="Informe de la simulación")
    pdf.ln(0.4)

    print_section(pdf, "Datos de la simulación", font_family, section_size, margin)

    pdf.cell(margin*2, 0, txt="Mercado: " + data_name)
    pdf.cell(margin, 0, txt="Fecha inicio: " + from_date)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Estrategia: " + file_name)
    pdf.cell(margin, 0, txt="Fecha final: " + to_date)

    print_section(pdf, "Resultados", font_family, section_size, margin)

    pdf.cell(margin*2, 0, txt="Inicial: %.2f\n" % initial_value)
    pdf.cell(margin, 0, txt="Trades+: %i\n" % trades_positives)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Final: %.2f\n" % final_value)
    pdf.cell(margin, 0, txt="Trades-: %i\n" % trades_negatives)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Ganancia(%%): %.2f\n" % percentage_profit)
    pdf.cell(margin, 0, txt="Avg trade: %.2f\n" % avg_trade)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Ganancias: %.2f\n" % net_profit)
    pdf.cell(margin, 0, txt="Avg profit: %.2f\n" % avg_profit_trade)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Max DD: %.2f\n" % maxdd)
    pdf.cell(margin, 0, txt="Avg loss: %.2f\n" % avg_loss_trade)

    pdf.ln(line_sep)

    pdf.cell(margin*2, 0, txt="Trades total: %i\n" % trades_total)
    pdf.cell(margin, 0, txt="Profit/Loss: %.2f\n\n\n" % avg_profit_loss)

    print_section(pdf, "Simulación", font_family, section_size, margin)

    execution_plot.plot_simulation(cerebro, file_name, data_name, from_date, to_date, size='big')

    image_path = ''
    pdf_path = ''

    if from_date==None or to_date==None:
        image_path = '../img/simulacion_' + file_name + '/' + data_name + '_' + file_name + 'big.png'
        pdf_path = '../reports/' + data_name + '_' + file_name + '.pdf'
    else:
        image_path = '../img/simulacion_' + file_name + '/' + data_name + '_' + from_date + '_' + to_date + '_' + file_name + 'big.png'
        pdf_path = '../reports/' + data_name + '_' + file_name + '_' + from_date + '_' + to_date + '.pdf'

    pdf.image(image_path, x=0.0, y=5.8, w=8.0)
    os.remove(image_path)

    create_folder_if_not_exists('../reports')
    pdf.output(pdf_path)
