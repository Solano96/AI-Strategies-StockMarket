# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font
from random import randint
import os.path as path
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_finance import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter
matplotlib.use('TkAgg')

import fix_yahoo_finance as yf

import datetime
import datetime as dt
from datetime import timedelta
import time
import threading

import pandas as pd
import numpy as np
import math
from datetime import timedelta
from sklearn.model_selection import cross_validate

import utils.indicators as indicators

from sklearn.model_selection import train_test_split

import itertools
import threading
import sys
from subprocess import Popen, PIPE

import backtrader as bt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import utils.model as model
import utils.func_utils as func_utils
import utils.myCerebro as myCerebro
import utils.myAnalyzer as myAnalyzer
import utils.testStrategyInteractive as testStrategyInteractive

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from numpy.random import seed

import tensorflow as tf

import threading

os.environ['KERAS_BACKEND'] = 'theano'
graph = tf.get_default_graph()

df = None
start_date = dt.datetime(2017, 1, 1)
end_date = dt.datetime(2018, 1, 1)

dates_list = None
showGrid = True

class STDText(Text):
    def __init__(self, parent):
        Text.__init__(self, parent)

    def write(self, stuff):
        self.insert("end", stuff)
        self.yview_pickplace("end")

    def flush(self):
        pass

def app():
    figure_w = 10
    figure_h = 4.5

    # initialise a window.
    root = Tk()
    root.style = ttk.Style()
    root.config(background='white')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))

    lab = Label(root, bg = 'white').pack()

    text_box = STDText(root)
    text_box.pack()

    sys.stdout = text_box

    plt.rc('xtick', labelsize=8)
    fig = plt.figure(figsize=(figure_w,figure_h),dpi=100)

    ax = fig.add_subplot(111)

    if showGrid:
        ax.grid(color='gray', linestyle='solid', lw=0.5, ls = '--')

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.05, right=0.95, hspace=0.0, wspace=0.0)

    graph1 = FigureCanvasTkAgg(fig, master=root)
    graph1.get_tk_widget().place(x=0, y=20)

    # Imprimir grafico chartista en pantalla
    def plotterChart():

        graph1 = FigureCanvasTkAgg(fig, master=root)
        graph1.get_tk_widget().place(x=0, y=20)

        setDate()
        ax.cla()

        if showGrid:
            ax.grid(color='gray', linestyle='solid', lw=0.5, ls = '--')

        #convert dates to datetime, then to float
        df2 = df.truncate(before=start_date, after=end_date)
        N = len(df2.index.strftime('%Y/%m/%d'))

        #convert dates to datetime, then to float
        date_val=mdates.date2num(df2.index.to_pydatetime())

        open_val = df2['Open'].values
        high_val = df2['High'].values
        low_val = df2['Low'].values
        close_val = df2['Close'].values
        volume_val = df2['Volume'].values
        # ohlc_data needs to be a sequence of sequences
        ohlc_data=zip(*[np.arange(N),open_val,high_val,low_val,close_val])

        candlestick_ohlc(ax, ohlc_data, width=.8, colorup='#46FF00', colordown='#FF0000')

        # Format x axis for dates
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, N - 1)
            return df2.index.strftime('%Y/%m/%d')[thisind]

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
        #fig.autofmt_xdate()
        graph1.draw()

    def setDate():
        global start_date
        global end_date
        start_date = dt.datetime(start_year.get(), start_month.get(), start_day.get())
        end_date = dt.datetime(end_year.get(), end_month.get(), end_day.get())

    def getDate(y, m, d):
        year = str(y)
        month = str(m)
        day = str(d)

        if m < 10:
            month = '0' + str(m)
        if d < 10:
            day = '0' + str(d)

        return year + '-' + month + '-' + day

    def loadData():
        global df
        global data_color
        global actual_option_data

        df = func_utils.getData(data_name.get())
        stock_name.set(data_name.get())

        plotterChart()

    def trainParallel():
        newthread = threading.Thread(target=trainFunction)
        newthread.start()

    def trainFunction():

        global df
        global dates_list
        global model
        global graph

        graph1 = FigureCanvasTkAgg(fig, master=root)
        graph1.get_tk_widget().place(x=0, y=20)

        with graph.as_default():
            commission = commission_value.get()

            gain = gain_value.get()
            loss = loss_value.get()
            n_day = nday_value.get()
            epochs = epochs_value.get()

            df = func_utils.add_features(df)
            df = func_utils.add_label(df, gain=gain, loss=loss, n_day = n_day, commission=commission)

            start_train_date = getDate(start_year_train.get(), start_month_train.get(), start_day_train.get())
            end_train_date = getDate(end_year_train.get(), end_month_train.get(), end_day_train.get())
            start_test_date = getDate(start_year_test.get(), start_month_test.get(), start_day_test.get())
            end_test_date = getDate(end_year_test.get(), end_month_test.get(), end_day_test.get())

            df_train, df_test, X_train, X_test, y_train, y_test = func_utils.split_df_date(df, start_train_date, end_train_date, start_test_date, end_test_date)

            # ------------ Normalizamos los datos ------------ #

            sc = StandardScaler()
            X_train = preprocessing.scale(X_train)
            X_test = preprocessing.scale(X_test)

            # Ponemos los datos en formato correcto para usarlos en keras
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # ------------ Obtenemos el modelo de predicción ------------ #

            print("Entrenando red neuronal...")

            clf = model.NeuralNetwork()
            clf.build_model(input_shape = (X_train.shape[1], 1))
            clf.train(X_train, y_train, epochs = epochs)


            # ------------ Comenzar la simulación ------------ #

            clf.init_memory(X_train[len(X_train)-15:len(X_train)], y_train[len(y_train)-15:len(y_train)])

            dates_list = df.index.strftime('%Y-%m-%d')

            print("Realizando simulación...")

            predictions = []

            #n_steps = 1
            n_steps = int(len(X_test)/4)

            for k in range(1,n_steps+1):
                # Creamos la instancia cerebro
                cerebro = myCerebro.MyCerebro()

                tam = (int)(k*1.0*len(X_test)/n_steps)
                start_to_predict = (int)((k-1)*1.0*len(X_test)/n_steps)
                #tam = k

                # Creamos una instancia de la clase TestStrategy
                ts = testStrategyInteractive.TestStrategy
                ts.X_test = X_test[0:tam]
                ts.y_test = y_test[0:tam]
                ts.model = clf
                ts.n_day = n_day
                ts.dates_list = df.index.strftime('%Y-%m-%d')
                ts.start_to_predict = start_to_predict
                ts.predictions = predictions

                # Añadimos la estrategia al cerebro
                cerebro.addstrategy(ts)

                # Añadimos los datos al cerebro
                data = bt.feeds.PandasData(dataname = df_test[0:tam])
                cerebro.adddata(data)

                # Fijamos el dinero inicial y la comisión
                cerebro.broker.setcash(cash_value.get())
                cerebro.broker.setcommission(commission=commission)

                result = cerebro.run()

                predictions =  result[0].predictions
                clf = result[0].model

                if k == n_steps:
                    fig_cerebro = cerebro.getFig(iplot=False)[0][0]
                else:
                    fig_cerebro = cerebro.getFig(iplot=False, start=max(tam-60, 0), end=tam)[0][0]

                fig_cerebro.set_figheight(figure_h)
                fig_cerebro.set_figwidth(figure_w)

                plt.subplots_adjust(top=0.98, bottom=0.1, left=0.05, right=0.95, hspace=0.0, wspace=0.0)

                graph1.figure = fig_cerebro
                graph1.draw()

            print("Simulación realizada con éxito...")

    fuente = font.Font(family="Helvetica",weight='bold')
    fuente2 = font.Font(family="Helvetica",size=14)
    fuente3 = font.Font(family="Helvetica",size=10, weight='bold')

    # Posicionar botones
    graph1.get_tk_widget().update()
    graph_width = graph1.get_tk_widget().winfo_width()+30
    graph_height = graph1.get_tk_widget().winfo_height()
    bwidth = 100
    bheight = 30

    sy = 30
    ygap = 1
    sy2 = 310

    stock_name = StringVar()
    stock_name.set('SAN')

    # Introducir Fecha
    start_day = IntVar()
    start_month = IntVar()
    start_year = IntVar()
    end_day = IntVar()
    end_month = IntVar()
    end_year = IntVar()

    # Fechas training
    #------------------------------------------------------
    start_day_train = IntVar()
    start_month_train = IntVar()
    start_year_train = IntVar()

    start_day_train.set(1)
    start_month_train.set(5)
    start_year_train.set(2017)

    start_train_d_text = ttk.Entry(root, textvariable=start_day_train)
    start_train_m_text = ttk.Entry(root, textvariable=start_month_train)
    start_train_y_text = ttk.Entry(root, textvariable=start_year_train)

    end_day_train = IntVar()
    end_month_train = IntVar()
    end_year_train = IntVar()

    end_day_train.set(1)
    end_month_train.set(5)
    end_year_train.set(2018)

    end_train_d_text = ttk.Entry(root, textvariable=end_day_train)
    end_train_m_text = ttk.Entry(root, textvariable=end_month_train)
    end_train_y_text = ttk.Entry(root, textvariable=end_year_train)
    #------------------------------------------------------

    # Fechas test
    #------------------------------------------------------
    start_day_test = IntVar()
    start_month_test = IntVar()
    start_year_test = IntVar()

    start_day_test.set(1)
    start_month_test.set(5)
    start_year_test.set(2018)

    start_test_d_text = ttk.Entry(root, textvariable=start_day_test)
    start_test_m_text = ttk.Entry(root, textvariable=start_month_test)
    start_test_y_text = ttk.Entry(root, textvariable=start_year_test)

    end_day_test = IntVar()
    end_month_test = IntVar()
    end_year_test = IntVar()

    end_day_test.set(1)
    end_month_test.set(5)
    end_year_test.set(2019)

    end_test_d_text = ttk.Entry(root, textvariable=end_day_test)
    end_test_m_text = ttk.Entry(root, textvariable=end_month_test)
    end_test_y_text = ttk.Entry(root, textvariable=end_year_test)
    #------------------------------------------------------

    cash_value = DoubleVar()
    cash_value.set(1000.0)
    cash_text = ttk.Entry(root, textvariable=cash_value)

    commission_value = DoubleVar()
    commission_value.set(0.001)
    commission_text = ttk.Entry(root, textvariable=commission_value)

    gain_value = DoubleVar()
    gain_value.set(0.1)
    gain_text = ttk.Entry(root, textvariable=gain_value)

    loss_value = DoubleVar()
    loss_value.set(0.03)
    loss_text = ttk.Entry(root, textvariable=loss_value)

    nday_value = IntVar()
    nday_value.set(15)
    nday_text = ttk.Entry(root, textvariable=nday_value)

    epochs_value = IntVar()
    epochs_value.set(100)
    epochs_text = ttk.Entry(root, textvariable=epochs_value)

    # Datos
    #-------------------------------------------------------------------------------------------------------------------------

    dataLabel = Label(root, text="Datos", font=fuente, fg = 'black', bg = 'white', borderwidth= 2,  relief="groove")
    getDataLabel = Label(root, text="Base datos: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)

    data_name = StringVar()
    data_name.set('SAN')
    data_text = ttk.Entry(root, textvariable=data_name)

    dataButton2 = Button(root, text="Cargar", command=loadData, bg="#007ABF", fg="white")

    fil = 0
    dataLabel.place(x=graph_width+bwidth*0, y=sy+bheight*ygap*fil, width=bwidth*2, height=bheight)
    fil+=1
    getDataLabel.place(x=graph_width+bwidth*0, y=sy+bheight*fil, width=bwidth, height=bheight)
    data_text.place(x=graph_width+bwidth*1, y=sy+bheight*ygap*fil, width=bwidth, height=bheight)
    fil+=1
    dataButton2.place(x=graph_width+bwidth*0, y=sy+bheight*ygap*fil, width=bwidth*2, height=bheight)
    #-------------------------------------------------------------------------------------------------------------------------

    # Gráfica
    #-------------------------------------------------------------------------------------------------------------------------
    start_day.set(1)
    start_month.set(5)
    start_year.set(2017)
    end_day.set(1)
    end_month.set(5)
    end_year.set(2019)

    graphLabel = Label(root, text="Gráfica", font=fuente, fg = 'black', bg = 'white', borderwidth = 2,  relief="groove")
    fromDate = Label(root, text="Desde: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    toDate = Label(root, text="Hasta: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)

    sd_text = ttk.Entry(root, textvariable=start_day)
    sm_text = ttk.Entry(root, textvariable=start_month)
    sy_text = ttk.Entry(root, textvariable=start_year)
    ed_text = ttk.Entry(root, textvariable=end_day)
    em_text = ttk.Entry(root, textvariable=end_month)
    ey_text = ttk.Entry(root, textvariable=end_year)

    chartButton = Button(root, text="Actualizar", command=plotterChart, bg='#007ABF', fg="white")

    fil1 = 4

    graphLabel.place(x=graph_width, y=sy+bheight*ygap*fil1, width=bwidth*2, height=bheight)
    fil1+=1

    fromDate.place(x=graph_width, y=sy+bheight*ygap*fil1, width=bwidth, height=bheight)
    sd_text.place(x=graph_width+bwidth+30*0, y=sy+bheight*ygap*fil1, width=30, height=bheight)
    sm_text.place(x=graph_width+bwidth+30*1, y=sy+bheight*ygap*fil1, width=30, height=bheight)
    sy_text.place(x=graph_width+bwidth+30*2, y=sy+bheight*ygap*fil1, width=40, height=bheight)
    fil1+=1

    toDate.place(x=graph_width, y=sy+bheight*ygap*fil1, width=bwidth, height=bheight)
    ed_text.place(x=graph_width+bwidth+30*0, y=sy+bheight*ygap*fil1, width=30, height=bheight)
    em_text.place(x=graph_width+bwidth+30*1, y=sy+bheight*ygap*fil1, width=30, height=bheight)
    ey_text.place(x=graph_width+bwidth+30*2, y=sy+bheight*ygap*fil1, width=40, height=bheight)
    fil1+=1

    chartButton.place(x=graph_width, y=sy+bheight*ygap*fil1, width=bwidth*2, height=bheight)

    # Simulador
    #-------------------------------------------------------------------------------------------------------------------------
    simulatorLabel = Label(root, text="Simulador", font=fuente, fg = 'black', bg = 'white', borderwidth= 2,  relief="groove")

    startTrainLabel = Label(root, text="Inicio Train: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    endTrainLabel = Label(root, text="Fin Train: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    startTestLabel = Label(root, text="Inicio Test: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    endTestLabel = Label(root, text="Fin Test: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)

    cashLabel = Label(root, text="Efectivo: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    commissionLabel = Label(root, text="Comisión: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    gainLabel = Label(root, text="Ganancia: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    lossLabel = Label(root, text="Pérdida: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    ndayLabel = Label(root, text="Num. Días: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)
    epochsLabel = Label(root, text="Iteraciones: ", font=fuente3, fg = 'black', bg = 'white', borderwidth= 2)

    trainButton = Button(root, text="Simular", command=trainParallel, bg="#007ABF", fg="white")
    #-------------------------------------------------------------------------------------------------------------------------

    fil2 = 0

    simulatorLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth*2, height=bheight)
    fil2+=1

    cashLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    cash_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    commissionLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    commission_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    gainLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    gain_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    lossLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    loss_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    ndayLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    nday_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    epochsLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    epochs_text.place(x=graph_width+bwidth, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    fil2+=1

    # Inicio Train
    startTrainLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    start_train_d_text.place(x=graph_width+30*0+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    start_train_m_text.place(x=graph_width+30*1+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    start_train_y_text.place(x=graph_width+30*2+bwidth, y=sy2+bheight*ygap*fil2, width=40 , height=30)
    fil2+=1

    # Fin Train
    endTrainLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    end_train_d_text.place(x=graph_width+30*0+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    end_train_m_text.place(x=graph_width+30*1+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    end_train_y_text.place(x=graph_width+30*2+bwidth, y=sy2+bheight*ygap*fil2, width=40 , height=30)
    fil2+=1

    # Inicio Test
    startTestLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    start_test_d_text.place(x=graph_width+30*0+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    start_test_m_text.place(x=graph_width+30*1+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    start_test_y_text.place(x=graph_width+30*2+bwidth, y=sy2+bheight*ygap*fil2, width=40 , height=30)
    fil2+=1

    # Fin Test
    endTestLabel.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth, height=30)
    end_test_d_text.place(x=graph_width+30*0+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    end_test_m_text.place(x=graph_width+30*1+bwidth, y=sy2+bheight*ygap*fil2, width=30 , height=30)
    end_test_y_text.place(x=graph_width+30*2+bwidth, y=sy2+bheight*ygap*fil2, width=40 , height=30)
    fil2+=1

    trainButton.place(x=graph_width, y=sy2+bheight*ygap*fil2, width=bwidth*2, height=bheight)
    fil2+=1

    text_box.place(x=50, y=graph_height+20, width=graph_width-130, height=bheight*6.5)

    root.mainloop()

if __name__ == '__main__':
    app()
