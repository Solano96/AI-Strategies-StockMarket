import backtrader as bt

class MyBuySell(bt.observers.BuySell):
        params = (('barplot', True), ('bardist', 0.05))
        plotlines = dict(
            buy=dict(marker='^', markersize=6.0, color='lime', fillstyle='full'),
            sell=dict(marker='v', markersize=6.0, color='red', fillstyle='full')
        )
