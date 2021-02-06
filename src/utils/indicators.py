import talib as ta

# Media movil
def moving_average(df, n):
	"""
	Add moving average indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: moving average period
	:return: dataframe with moving average of n period added as a column
	"""
	if n != 1:
		df['MA_' + str(n)] = ta.EMA(df['Close'], timeperiod=n)
	else:
		df['MA_' + str(n)] = df['Close']

	return df

# Media exponencial
def exponential_moving_average(df, n):
	"""
	Add exponential moving average indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: exponential moving average period
	:return: dataframe with exponential moving average of n period added as a column
	"""
	df['EMA_' + str(n)] = ta.EMA(df['Close'], timeperiod=n)
	return df

# Momento
def momentum(df, n):
	"""
	Add momentum indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: momentum period
	:return: dataframe with momentum of n period added as a column
	"""
	df['Momentum_' + str(n)] = ta.MOM(df['Close'], timeperiod=n)
	return df

# Indice de fuerza relativa RSI
def relative_strength_index(df, n):
	"""
	Add rsi indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: rsi period
	:return: dataframe with rsi of n period added as a column
	"""
	df['RSI_' + str(n)] = ta.RSI(df['Close'], timeperiod=n)
	return df

# Desviacion tipica
def standard_deviation(df, n):
	"""
	Add standard deviation indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: standard deviation period
	:return: dataframe with standard deviation of n period added as a column
	"""
	df['STD_' + str(n)] = ta.STDDEV(df['Close'], timeperiod=n)
	return df

# ROC
def rate_of_change(df, n):
	"""
	Add rate of change indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param n: rate of change period
	:return: dataframe with rate of change of n period added as a column
	"""
	df['ROC_' + str(n)] = ta.ROC(df['Close'], timeperiod=n)
	return df

# MACD e histograma
def moving_average_CD(df, s, f):
	"""
	Add moving average CD indicator as a column in dataframe df
	:param df: dataframe with historical prices
	:param f: fast period
	:param s: slow period
	:return: dataframe with moving average CD and histogram as a columns
	"""
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
