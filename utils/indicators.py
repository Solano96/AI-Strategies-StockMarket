import talib as ta

# Media movil
def moving_average(df, n):
	df['MA_' + str(n)] = ta.MA(df['Close'], timeperiod=n)
	return df

# Media exponencial
def exponential_moving_average(df, n):
    df['EMA_' + str(n)] = ta.EMA(df['Close'], timeperiod=n)
    return df

# Momento
def momentum(df, n):
    df['Momentum_' + str(n)] = ta.MOM(df['Close'], timeperiod=n)
    return df

# Indice de fuerza relativa RSI
def relative_strength_index(df, n):
	df['RSI_' + str(n)] = ta.RSI(df['Close'], timeperiod=n)
	return df

# Desviacion tipica
def standard_deviation(df, n):
    df['STD_' + str(n)] = ta.STDDEV(df['Close'], timeperiod=n)
    return df

# ROC
def rate_of_change(df, n):
    df['ROC_' + str(n)] = ta.ROC(df['Close'], timeperiod=n)
    return df

# MACD e histograma
def moving_average_CD(df, s, f):
    macd, macdsignal, macd_hist = ta.MACD(df['Close'], fastperiod=f, slowperiod=s, signalperiod=9)
    df['MACD_'+str(s)+'_'+str(f)] = macd
    df['MACD_HIST_'+str(s)+'_'+str(f)] = macd_hist
    return df

# Oscilador estocastico lento
def stochastic(df, n):
    slowk, slowd = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=n, slowk_period=n, slowd_period=n)
    df['STOCH_SLOW_K_'+str(n)] = slowk
    df['STOCH_SLOW_D_'+str(n)] = slowd
    return df

# Oscilador estocastico rapido
def stochastic_fast(df, n):
    fastk, fastd = ta.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=n, fastd_period=n, fastd_matype=0)
    df['STOCH_FAST_K_'+str(n)] = fastk
    df['STOCH_FAST_D_'+str(n)] = fastd
    return df
