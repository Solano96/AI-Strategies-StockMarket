# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys
import backtrader as bt
from fpdf import FPDF
import webbrowser

import src.strategies_execution.execution_plot as execution_plot


def create_folder_if_not_exists(folder_name):
    """
    Create folder inside img folder
    :param folder_name: folder name
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def printAnalysis(info, params, metrics, training_params=None):
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

    create_folder_if_not_exists('./resultados')

    file_name = info['Estrategia']

    f = open ('./resultados/results.log','a')

    f.write('------------------------------------\n\n')

    for key, value in info.items():
        f.write("{0}: {1}\n".format(key, value))

    if len(params) > 0:
        f.write("\n\n")

        for key, value in params.items():
            f.write("{0}: {1}\n".format(key, value))

    if training_params != None:
        f.write("\n\n")

        for key, value in training_params.items():
            f.write("{0}: {1}\n".format(key, value))

    f.write("\n\n")

    for key, value in metrics.items():
        if isinstance(value, float):
            value = round(value, 2)
        f.write("{0}: {1}\n".format(key, value))

    f.close()


def print_section(pdf, text_section, font_family, section_size, margin):
    pdf.ln(0.2)
    #pdf.set_text_color(0,75,126)
    pdf.set_font(font_family, style='B', size=section_size)
    pdf.cell(margin, 0, txt=text_section)
    pdf.set_font(font_family, style='', size=10)
    pdf.set_text_color(0,0,0)
    pdf.ln(0.3)


def print_dict(pdf, my_dict, margin, line_sep):
    for item, i in zip(my_dict.items(), range(0, len(my_dict))):
        key, value = item

        if isinstance(value, float):
            value = round(value, 2)

        if i%2 == 0:
            pdf.cell(margin*2, 0, txt="{0}: {1}".format(key, value))
        else:
            pdf.cell(margin, 0, txt="{0}: {1}".format(key, value))

        if i%2 != 0 or i == len(my_dict)-1:
            pdf.ln(line_sep)


def printAnalysisPDF(cerebro, info, params, metrics, training_params=None):
    '''
    Function to generate a report in PDF format.
    :param cerebro: backtrader engine (necesary for plot)
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

    file_name = info['Estrategia']
    data_name = info['Mercado']
    from_date = info['Fecha inicial']
    to_date = info['Fecha final']

    pdf = FPDF(unit='in')
    effective_page_width = pdf.w - 2*pdf.l_margin
    sep = effective_page_width/4.0
    section_size = 12
    font_family = 'Helvetica'

    pdf.add_page()
    pdf.set_font(font_family, style='B', size=20)

    pdf.set_line_width(0.02)
    #pdf.set_draw_color(0,75,126)
    #pdf.set_draw_color(200,10,10)
    pdf.line(0.4, 0.5, 7.8, 0.5)
    pdf.line(0.4, 1.4, 7.8, 1.4)
    pdf.line(0.4, 11.2, 7.8, 11.2)

    margin = sep
    line_sep = 0.2
    pdf.ln(0.6)
    pdf.cell(margin, 0, txt="Informe de la simulación")
    pdf.ln(0.6)

    # Print simulation info
    print_section(pdf, "Datos de la simulación", font_family, section_size, margin)
    print_dict(pdf, info, margin, line_sep)

    # Print more information about simulation
    if len(params) > 0:
        print_section(pdf, "Más información", font_family, section_size, margin)
        print_dict(pdf, params, margin, line_sep)

    if training_params != None:
        print_section(pdf, "Parámetros del entrenamiento", font_family, section_size, margin)
        print_dict(pdf, training_params, margin, line_sep)

    # Print simulation metrics
    print_section(pdf, "Resultados", font_family, section_size, margin)
    print_dict(pdf, metrics, margin, line_sep)

    print_section(pdf, "Simulación", font_family, section_size, margin)

    # Image with 800x500 pixels (8,5)
    image_path = execution_plot.plot_simulation(cerebro, file_name, data_name, from_date, to_date,
                                                size=(10,6), style='line')

    # PDF path
    pdf_path = './reports/' + data_name + '_' + file_name + '_' + from_date + '_' + to_date + '.pdf'

    pdf.image(image_path, x=0.0, y=pdf.get_y(), w=8.0)
    os.remove(image_path)


    create_folder_if_not_exists('./reports')
    pdf.output(pdf_path)
    webbrowser.open_new_tab(pdf_path)
