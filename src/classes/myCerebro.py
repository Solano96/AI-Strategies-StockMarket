import backtrader as bt


class MyCerebro(bt.Cerebro):

    def getFig(self, plotter=None, numfigs=1, iplot=True, start=None, end=None,
         width=16, height=9, dpi=300, tight=True, use=None,
         **kwargs):

        if self._exactbars > 0:
            return

        if not plotter:
            from backtrader import plot
            import matplotlib
            matplotlib.use('TkAgg')
            if self.p.oldsync:
                plotter = plot.Plot_OldSync(**kwargs)
            else:
                plotter = plot.Plot(**kwargs)

        figs = []
        for stratlist in self.runstrats:
            for si, strat in enumerate(stratlist):
                rfig = plotter.plot(strat, figid=si * 100,
                                    numfigs=numfigs, iplot=iplot,
                                    start=start, end=end, use=use)

                figs.append(rfig)

        return figs
