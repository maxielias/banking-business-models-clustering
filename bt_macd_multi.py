from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import random

import backtrader as bt
import backtrader.indicators as btind

import logging

logging.basicConfig(
    filename="freqtrade/finamom/Futures/report/report_multi.log",
    level=logging.DEBUG,
    format="%(message)s"#"%(asctime)s:%(levelname)s:%(message)s"
    )

'''class BitmexComissionInfo(bt.CommissionInfo):
    params = (
        ("commission", 0.00075),
        ("mult", 1.0),
        ("margin", None),
        ("commtype", None),
        ("stocklike", False),
        ("percabs", False),
        ("interest", 0.0),
        ("interest_long", False),
        ("leverage", 1.0),
        ("automargin", False),
    )
def getsize(self, price, cash):
    """Returns fractional size for cash operation @price"""
    return self.p.leverage * (cash / price)'''

class MACD(bt.Strategy):
    params = (
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
        ('ema200', 200),
        # Percentage of portfolio for a trade. Something is left for the fees
        # otherwise orders would be rejected
        ('portfolio_frac', 0.02),
        ('take_profit', 0.007),
        ('stop_loss', 0.2),
        #("dca1_leverage", 1),
        #("dca2_leverage", 2),
        ('dca_price_var1', 0.005),
        ('dca_price_var2', 0.05),
        #("dca1_counter", 0.0),
        #("buy_price_adjust", 0.000),
        #("buy_limit_adjust", 0.000),
        #("buy_stop_adjust", 0.000),
        #("p1", 5),
        #("p2", 15),
        ('limit', 0.005),
        ('limdays', 3),
        ('limdays2', 1000),
        ('hold', 100000),
        ('oneplot', True)
        #("usebracket", False),  # use order_target_size
        #("switchp1p2", False),  # switch prices of order1 and order2
        #("trail", False),
        #("buy_limit", False),
        #("oco1oco2", False),  # False - use order1 as oco for order3, else order2
        #("do_oco", True),  # use oco or not
    )

    def __init__(self):

        '''
        For an official backtrader blog on this topic please take a look at:
        https://www.backtrader.com/blog/posts/2017-04-09-multi-example/multi-example.html
        oneplot = Force all datas to plot on the same master.
        '''

        self.ind_dict = dict()
        #self.ind_dict_30m = dict()
        self.ord = dict()
        self.holding = dict()
        self.holdstart = dict ()

        #self.data5m = self.datas
        #self.data5m = self.datas[::2]
        #self.data30m = self.datas[1::2]

        # Data for 5m candlesticks:
        for i, df5m in enumerate(self.datas):

            self.ind_dict[df5m] = dict()

            self.val_start = self.broker.get_cash()  # keep the starting cash
            #self.size = None
            #self.order = None
            #self.long = None
            #self.buy_price = 0
            #self.dca_count = 0

            # List of orders in bt documents
            #self.orefs = list()

            self.buy_price_executed_order_list = list()
            self.sell_price_executed_order_list = list()
            self.stop_loss_price_executed_order_list = list()
            self.take_profit_price_executed_order_list = list()

            self.ind_dict[df5m]['macd'] =  bt.ind.MACD(
                                                df5m.close,
                                                period_me1=self.p.macd1,
                                                period_me2=self.p.macd2,
                                                period_signal=self.p.macdsig
                                            )
            
            self.ind_dict[df5m]['mcross'] = bt.ind.CrossOver(
                                                self.ind_dict[df5m]['macd'].macd,
                                                self.ind_dict[df5m]['macd'].signal
                                            )

            if i > 0: #Check we are not on the first loop of data feed:

                if self.p.oneplot == True:
                    df5m.plotinfo.plotmaster = self.datas[0]

        # Data for 30m candlesticks:
        #for i, df30m in enumerate(self.data30m):    
            
        #    self.ind_dict_30m[df30m]['ema'] = bt.ind.EMA(
        #                    df30m.close,
        #                    period=self.p.ema200
        #                )

        #    self.ind_dict[df30m]['ema200_30m_cross'] = bt.ind.CrossOver(
        #                                df30m.close,
        #                                self.ind_dict_30m[df30m]['ema']
        #                            )
            
    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """
        dt, dn = self.datetime.datetime(0), order.data._name

        print('{} {} Order {} Status {}'.format(
            dt, dn, order.ref, order.getstatusname())
        )
        
        if order.alive():
            print("Order Alive")
            logging.debug("Order Alive")

        if not order.alive():
            df5m_orders = self.ord[order.data]
            idx = df5m_orders.index(order)
            df5m_orders[idx] = None
            print('-- No longer alive {} Ref'.format(df5m_orders[idx]))

            if all(x is None for x in df5m_orders):
                df5m_orders[:] = []  # empty list - New orders allowed

        if order.status == order.Completed:
            if order.isbuy():              
                self.holdstart[self.df5m] = len(self.df5m)
                order_side = "Buy"
                print(
                    f"{order_side} Order Completed -  Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Date: {self.data0.datetime.datetime(0)} "
                    f"Take Profit Price: {order.executed.price * (1 + (self.p.take_profit))} "
                    f"Stop Loss Price: {order.executed.price * (1- (self.p.stop_loss))} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                    f"Order Ref. {self.ord} "
                )
                logging.debug(
                        f"{order_side} Order Completed -  Size: {order.executed.size} "
                        f"@Price: {order.executed.price} "
                        f"Date: {self.data0.datetime.datetime(0)} "
                        f"Take Profit Price: {order.executed.price * (1 + (self.p.take_profit))} "
                        f"Stop Loss Price: {order.executed.price * (1- (self.p.stop_loss))} "
                        f"Value: {order.executed.value:.2f} "
                        f"Comm: {order.executed.comm:.6f} "
                        f"Order Ref. {self.ord} "
                )

                self.buy_price_executed_order_list = [order.executed.price]
                self.stop_loss_price_executed_order_list = [order.executed.price * (1 - (self.p.stop_loss))]
                self.take_profit_price_executed_order_list = [order.executed.price * (1 + (self.p.take_profit))]


            else:
                order_side = "Sell"
                print(
                    f"{order_side} Order Completed -  Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Date: {self.data0.datetime.datetime(0)} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                    f"Order Ref. {self.orefs} "
                    f"Order Status: {order.status} - {order.getstatusname()}"
                )
                logging.debug(
                    f"{order_side} Order Completed -  Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Date: {self.data0.datetime.datetime(0)} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                    f"Order Ref. {self.orefs} "
                    f"Order Status: {order.status} - {order.getstatusname()}"
                )

                self.sell_price_executed_order_list = [order.executed.price]


        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            print(f"{order_side} Order Canceled/Margin/Rejected")
            logging.debug(f"{order_side} Order Canceled/Margin/Rejected")

        df5m_orders = None  # indicate no order pending

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""
        
        dt, dn = self.datetime.datetime(0), df5m._name
        
        if not trade.isclosed:
            return
        print(f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}", f"Position {self.position.upopened}")
        #logging.debug(f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}", f"Position {self.position.upopened}")

        """ Calculate the actual returns """
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        val_end = self.broker.get_value()
        
        print(
            f"ROI: {100.0 * self.roi:.2f}%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )
        logging.debug(
            f"ROI: {100.0 * self.roi:.2f}%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )
    
    def stop(self):
        """ Calculate the actual returns """
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        val_end = self.broker.get_value()
        
        print(
            f"ROI: {100.0 * self.roi:.2f}%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )
        logging.debug(
            f"ROI: {100.0 * self.roi:.2f}%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )
    
    def next(self):
        
        #for i, (df5m, df30m) in enumerate(zip(self.datas5m, self.datas30m)):
        for i, df5m in enumerate(self.datas):
            
            dt, dn = self.datetime.date(), df5m._name
        
            # Start
            #close_price_vs_ema200 = "Close Price > EMA 200" if df5m.close[0] > self.ind_dict[df30m]['ema'][0] else "Close Price <= EMA 200"

            #if self.order:
            #'''Chequear como reescribir esta condición'''
            if not self.ord.get(df5m, None):

                return # pending orders do nothing

            print(
                #f"Pending order execution. Waiting in orderbook. "
                f"DateTime {self.df5m.datetime.datetime(0)}, "
                f"Close 5m: {self.df5m[0]:.2f}, MACDcross {self.ind_dict['mcross'][0]}" # - {close_price_vs_ema200}, MACDcross {self.ind_dict['mcross'][0]}, "
                f"Position Size {self.getposition(df5m).size}"
            )
            logging.debug(
                #f"Pending order execution. Waiting in orderbook. "
                f"DateTime {self.df5m.datetime.datetime(0)}, "
                f"Close 5m: {self.df5m[0]:.2f}, MACDcross {self.ind_dict['mcross'][0]}" # - {close_price_vs_ema200}, MACDcross {self.ind_dict['mcross'][0]}, "
                f"Position Size {self.getposition(df5m).size}"
            )

            if not self.getposition(df5m).size:
                
                if (self.ind_dict['mcross'][0] > 0.0): # and (df5m.close[0] > self.ind_dict['ema'][0]): # conditions for entering the market
                    
                    print("Time to Buy.")
                    logging.debug("Time to Buy.")

                    # Data needed for the self.buy()
                    self.buy_price = df5m.close[0] # price of the 5m candle after triggering singal

                    self.long = self.buy(
                                        data=df5m,
                                        price=self.buy_price,
                                        size=self.broker.get_cash() / df5m.close * self.p.portfolio_frac,
                                        exectype=bt.Order.Limit
                                    )

                    # Data needed for self.sell() and dca position
                    self.order = self.long

                    # Tuples needed to reference values to another functions and instances
                    self.ordref = [self.ord.get(df5m, None)] # order reference number

                    print(
                        f"Order Ref = {self.ord}, "
                        f"Long Size = {self.long.price}, " # Long price is the same as order price
                        f"Long Size = {self.long.size}, "
                        f"Cash = {self.broker.get_cash()}, "
                        f"Value = {self.broker.get_value()}"
                        f"Percentage Invested = {self.p.portfolio_frac}, " # % per order
                    )

                    loggin.debug(
                            f"Order Ref = {self.ord}, "
                            f"Long Size = {self.long.price}, " # Long price is the same as order price
                            f"Long Size = {self.long.size}, "
                            f"Cash = {self.broker.get_cash()}, "
                            f"Value = {self.broker.get_value()}"
                            f"Percentage Invested = {self.p.portfolio_frac}, " # % per order
                    )

            elif self.getposition(df5m).size:  # in the market
                
                # sell with take_profit or stop_loss:
                if df5m.close[0] <= (self.position.price * (1 - self.p.stop_loss)):
                    
                    print("You Should Sell for Loss.")
                    logging.debug("You Should Sell for Loss.")
                    self.order = self.sell(
                                        data=df5m,
                                        price=(self.position.price * (1 - self.p.stop_loss)),
                                        size=self.getposition(df5m).size,
                                        exectype=bt.Order.Limit
                                    )

                    self.ordref= [self.ord.get(df5m, None)]

                    print(
                        f"{self.ord}, "
                        f"Stop Loss Position Price = {self.position.price * (1 - self.p.stop_loss)}, "
                        f"Close = {df5m.close[0]}, "
                        f"DateTime {self.df5m.datetime.datetime(0)}, "
                        f"Position Price = {self.position.price}, "
                        f"Position Size = {self.position.size}, "
                        f"Cash = {self.broker.get_cash()}, "
                        f"Value = {self.broker.get_value()}, "
                        f"Percentage Invested = {self.p.portfolio_frac}"
                    )

                    logging.debug(
                        f"{self.ord}, "
                        f"Stop Loss Position Price = {self.position.price * (1 - self.p.stop_loss)}, "
                        f"Close = {df5m.close[0]}, "
                        f"DateTime {self.df5m.datetime.datetime(0)}, "
                        f"Position Price = {self.position.price}, "
                        f"Position Size = {self.position.size}, "
                        f"Cash = {self.broker.get_cash()}, "
                        f"Value = {self.broker.get_value()}, "
                        f"Percentage Invested = {self.p.portfolio_frac}"
                    )

                    self.dca_count = 0

                elif df5m.close[0] >= self.position.price * (1 + self.p.take_profit):
                    
                    print("You Should Sell for Profit.")
                    logging.debug("You Should Sell for Profit.")
                    self.order = self.sell(
                                        price=(self.position.price * (1 + self.p.take_profit)),
                                        size=self.getposition(df5m).size,
                                        exectype=bt.Order.Limit
                                    )

                    self.ordref = [self.ord.get(df5m, None)]

                    print(
                        f"{self.ord}, "
                        f"Take Profit Price = {self.position.price * (1 + self.p.take_profit)}, "
                        f"Close = {df5m.close[0]}, "
                        f"DateTime {self.df5m.datetime.datetime(0)}, "
                        f"Position Price = {self.position.price}, "
                        f"Position Size = {self.position.size}, "
                        f"Cash = {self.broker.get_cash()}, "
                        f"Value = {self.broker.get_value()}, "
                        f"Percentage Invested = {self.p.portfolio_frac}"
                    )

                    logging.debug(
                        f"{self.ord}, "
                        f"Take Profit Price = {self.position.price * (1 + self.p.take_profit)}, "
                        f"Close = {df5m.close[0]}, "
                        f"DateTime {self.df5m.datetime.datetime(0)}, "
                        f"Position Price = {self.position.price}, "
                        f"Position Size = {self.position.size}, "
                        f"Cash = {self.broker.get_cash()}, "
                        f"Value = {self.broker.get_value()}, "
                        f"Percentage Invested = {self.p.portfolio_frac}"
                    )

                    self.dca_count = 0
                
                # if the difference between the start of the backtest and the trade notification is bigger than the hold limit parameter
                elif (len(self) - self.holdstart) >= self.p.hold:

                    pass
                
                else:
                    # If there is a limit to the holding time this condition is true if holding time < holding limit time
                    print("Still in the Market Condition: "+str(len(self))+" - "+str(self.holdstart)+" = "+str(len(self) - self.holdstart))#+" : "+str(self.p.hold)+"")
                    logging.debug("Still in the Market Condition: "+str(len(self))+" - "+str(self.holdstart)+" = "+str(len(self) - self.holdstart))#+" : "+str(self.p.hold)+"")
                    
                    if self.position.size > 0:
                    
                        print(f"Position Size > 0 ---> {self.position.size}")
                        logging.debug(f"Position Size > 0 ---> {self.position.size}")

                        if self.data0.close[0] > self.position.price * (1 - self.p.dca_price_var1) and self.dca_count == 0:
                            
                            print("No DCA Yet")
                            logging.debug("No DCA Yet")

                        elif self.data0.close[0] <= self.position.price * (1 - self.p.dca_price_var1) and self.dca_count == 0:
                            
                            self.buy_price = self.data0.close[0] # price of the 5m candle after triggering signal 
                            
                            self.long = self.buy(
                                price=self.position.price * (1 - self.p.dca_price_var1),
                                size=self.getposition(df5m).size,
                                exectype=bt.Order.Limit,
                                )

                            # Data needed for self.sell() and dca position
                            self.order = self.long 

                            # Tuples needed to reference values to another functions and instances
                            self.ordref = [self.ord.get(df5m, None)] # order reference number

                            # DCA counter == 1
                            self.dca_count = 1

                            print("DCA 1 About to be Executed")
                            print(
                                f"GetPosition {self.getposition(self.data0).size}, "
                                f"Position.Price {self.position.price}"
                            )
                            logging.debug("DCA 1 About to be Executed")
                            logging.debug(
                                f"GetPosition {self.getposition(self.data0).size}, "
                                f"Position.Price {self.position.price}"
                            )

                        elif self.data0.close[0] <= self.position.price * (1 - self.p.dca_price_var2) and self.dca_count == 1:
                            
                            self.buy_price = self.data0.close[0] # price of the 5m candle after triggering signal 
                            #self.dca1_position = self.position_size_list[0]
                            
                            self.long = self.buy(
                                price=self.position.price * (1 - self.p.dca_price_var2),
                                size=self.getposition(self.data0).size,
                                exectype=bt.Order.Limit,
                                )

                            # Data needed for self.sell() and dca position
                            self.order = self.long 

                            # Tuples needed to reference values to another functions and instances
                            self.orefs = [self.order.ref] # order reference number

                            # DCA counter == 2
                            self.dca_count = 2

                            print("DCA 2 About to be Executed")
                            print(
                                f"GetPosition {self.getposition(self.data0).size}, "
                                f"Position.Price {self.position.price}"
                                )
                            logging.debug("DCA 2 About to be Executed")
                            logging.debug(
                                f"GetPosition {self.getposition(self.data0).size}, "
                                f"Position.Price {self.position.price}"
                            )
                        
                        else:

                            print("DCA 1 Used Already") if self.dca_count == 1 else print("DCA 1 and 2 Used Already")
                            print(
                                f"Get Position {self.getposition(self.data0).size}, "
                                f"Price {self.position.price}, "
                                f"Price {self.position.price * (1 - self.p.stop_loss)}, "
                                f"Price {self.position.price * (1 + self.p.take_profit)}"
                            )
                            logging.debug("DCA 1 Used Already") if self.dca_count == 1 else print("DCA 1 and 2 Used Already")
                            logging.debug(
                                f"Get Position {self.getposition(self.data0).size}, "
                                f"Price {self.position.price}, "
                                f"Price {self.position.price * (1 - self.p.stop_loss)}, "
                                f"Price {self.position.price * (1 + self.p.take_profit)}"
                            )
                            
                            pass

            else:

                print("WHAT")
                logging.debug("WHAT")

#cerebro = bt.Cerebro(cheat_on_open=True)
cerebro = bt.Cerebro()

# Amount of starting cash
cerebro.broker.set_cash(1000)

# Data feeds list (5min candles)
df_list = [
    ('freqtrade/finamom/Futures/datas/snxusdt_futures_5m.csv', 'SNXUSDT_5m'),    
    ('freqtrade/finamom/Futures/datas/zecusdt_futures_5m.csv', 'ZECUSDT_5m'),  
    ('freqtrade/finamom/Futures/datas/zilusdt_futures_5m.csv', 'ZILUSDT_5m'),      
]

#class GenericDataFeed(bt.feeds.GenericCSVData):

#    params = (
#        ('nullvalue', float('Nan')),
#        ('dtformat', '%Y-%m-%d %H:%M:%S'),
#        ('timeframe', Minutes)
#        ('compression', 5)
#        ('fromdate', datetime.datetime(2020, 11, 5, 0, 0, 0)),
#        ('todate', datetime.datetime(2020, 11, 15, 0, 0, 0)),
#        ('datetime', 1),
#        ('time', -1),
#        ('high', 2),
#        ('low', 3),
#        ('open', 4),
#        ('close', 5),
#        ('volume', 6),
#        ('openinterest', -1)
#    )

for i in range(len(df_list)):

    data = bt.feeds.GenericCSVData(
                                    dataname=df_list[i][0],
                                    #nullvalue=0.0,
                                    dtformat="%Y-%m-%d %H:%M:%S",
                                    timeframe=bt.TimeFrame.Ticks,
                                    compression=5,
                                    #fromdate=datetime.datetime(2020, 1, 1, 0, 0, 0),
                                    #todate=datetime.datetime(2020, 2, 1, 0, 0),
                                    datetime=1,
                                    high=2,
                                    low=3,
                                    open=4,
                                    close=5,
                                    volume=6,
                                    openinterest=-1,
            )
    #cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.adddata(data, name=df_list[i][1])
    
    #if i % 2 == 0 or 1 == 0:
    #    data = GenericDataFeed(dataname=df_list[i][0])
    #    cerebro.resampledata(dataname=data, timeframe=bt.TimeFrame.Minutes, compression=5)
    #    data = GenericDataFeed(dataname=df_list[i+1][0])
    #    cerebro.resampledata(dataname=data, timeframe=bt.TimeFrame.Minutes, compression=30)
    
    #else:
    #    pass
        
# cerebro.adddata(data, name=df_list[i][1])

# Add strategy
cerebro.addstrategy(MACD)

# Add comission (BitmexComissionInfo)
#cerebro.broker.addcommissioninfo(BitmexComissionInfo())
#cerebro.broker = bt.brokers.BackBroker(slip_fixed=0.0)
cerebro.broker.setcommission(commission=0.003)

# Add TimeReturn Analyzers to benchmark data
cerebro.addanalyzer(
    bt.analyzers.TimeReturn, _name="alltime_roi", timeframe=bt.TimeFrame.NoTimeFrame
)

cerebro.addanalyzer(
    bt.analyzers.TimeReturn,
    data=data,
    _name="benchmark",
    timeframe=bt.TimeFrame.NoTimeFrame,
)

# Execute
results = cerebro.run()
#results = cerebro.run(runonce=False)
st0 = results[0]

#logging.debug(results)
#logging.debug(st0)

alyzers_list = list()
results_list = list()

for alyzer in st0.analyzers:
    alyzer.print()
    alyzers_list.append(alyzer)

#print(alyzers_list)
#logging.debug(alyzers_list)

cerebro.plot(iplot=False, style="bar")