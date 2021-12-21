from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import random
import pandas as pd

import backtrader as bt
import backtrader.indicators as btind

import logging

tp_list = [0.003, 0.005, 0.007, 0.01]
dca_list = {'dca1':[0.005, 0.01, 0.05, 0.1], 'dca2':[0.05, 0.1, 0.15, 0.2]}
ticker_list = ['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT',
                 'LINKUSDT', 'XLMUSDT', 'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT', 'BNBUSDT',
                 'ATOMUSDT', 'ONTUSDT', 'IOTAUSDT', 'BATUSDT', 'VETUSDT', 'NEOUSDT', 'QTUMUSDT', 'IOSTUSDT',
                 'THETAUSDT', 'ALGOUSDT', 'ZILUSDT', 'KNCUSDT', 'ZRXUSDT', 'COMPUSDT', 'OMGUSDT', 'DOGEUSDT', 
                 'SXPUSDT', 'KAVAUSDT', 'BANDUSDT', 'RLCUSDT', 'WAVESUSDT', 'MKRUSDT', 'SNXUSDT', 'DOTUSDT', 
                 'DEFIUSDT', 'YFIUSDT', 'BALUSDT', 'CRVUSDT', 'TRBUSDT', 'YFIIUSDT', 'RUNEUSDT', 'SUSHIUSDT', 
                 'SRMUSDT', 'BZRXUSDT', 'EGLDUSDT', 'SOLUSDT', 'ICXUSDT', 'STORJUSDT', 'BLZUSDT', 'UNIUSDT',
                 'AVAXUSDT', 'FTMUSDT', 'HNTUSDT', 'ENJUSDT', 'FLMUSDT', 'TOMOUSDT', 'RENUSDT', 'KSMUSDT', 
                 'NEARUSDT', 'AAVEUSDT', 'FILUSDT', 'RSRUSDT', 'LRCUSDT', 'MATICUSDT', 'OCEANUSDT', 'CVCUSDT', 
                 'BELUSDT', 'CTKUSDT', 'AXSUSDT', 'ALPHAUSDT', 'ZENUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 
                 'BTCBUSD', 'AKROUSDT', 'DOTECOUSDT', 'CHZUSDT', 'SANDUSDT', 'ANKRUSDT', 'LUNAUSDT', 'BTSUSDT', 
                 'LITUSDT', 'UNFIUSDT', 'DODOUSDT']

for ticker in ticker_list:
#for tp in tp_list:

    coin_name = ticker
    dca1 = 0.005
    dca2 = 0.05
    dca3 = 0.18

    for tp in tp_list:
    #for dca1, dca2 in zip(dca_list['dca1'], dca_list['dca2']):

        logging.basicConfig(
            filename='freqtrade/finamom/Futures/report/report_'+ticker+'.log',
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
                ("macd1", 12),
                ("macd2", 26),
                ("macdsig", 9),
                ("ema200", 200),
                ("rsi_ema", 14),
                ("rsi_sma", 14),
                ("boll", 20),
                ("devfactor", 2),
                # Percentage of portfolio for a trade. Something is left for the fees
                # otherwise orders would be rejected
                ("portfolio_frac", 0.02),
                ("take_profit", tp),# 0.005),
                ("stop_loss", 0.3),
                #("dca1_leverage", 1),
                #("dca2_leverage", 2),
                ("dca_price_var1", dca1),
                ("dca_price_var2", dca2),
                ("dca_price_var3", dca3),
                #("dca1_counter", 0.0),
                #("buy_price_adjust", 0.000),
                #("buy_limit_adjust", 0.000),
                #("buy_stop_adjust", 0.000),
                #("p1", 5),
                #("p2", 15),
                ("limit", 0.005),
                ("limdays", 3),
                ("limdays2", 1000),
                ("hold", 100000),
                ("oneplot", True)
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

                self.val_start = self.broker.get_cash()  # keep the starting cash
                self.size = None
                self.order = None
                self.long = None
                self.buy_price = 0
                self.dca_count = 0

                # List of orders in bt documents
                self.orefs = list()
                self.buy_price_executed_order_list = list()
                self.sell_price_executed_order_list = list()
                self.stop_loss_price_executed_order_list = list()
                self.take_profit_price_executed_order_list = list()

                # Data needed for profit statistics
                self.holding_time = list()
                self.win_holding_time = list()
                self.lost_holding_time = list()
                self.win_counter = 0
                self.lost_counter = 0
                self.trades_counter = 0
                self.profit_sum = 0
                self.loss_sum = 0
                self.loss = list()
                self.profit = list()
                self.pnl_evol = list()

                # Data lists for excel
                #datetime_list = []
                
                self.datetime_list = list()
                self.close_price_list = list()
                self.dca_count_list = list()

                '''self.macd_macd_list = list()
                self.macd_signal_list = list()
                self.macd_cross_list = list()
                self.rsi_ema_list = list()
                self.rsi_sma_list = list()
                self.boll_mid_list = list()
                self.boll_bot_list = list()
                self.boll_top_list = list()
                self.sell_order_take_profit_price= list()
                self.sell_order_stop_loss= list()'''

                self.buy_order_datetime = list() 
                self.buy_order_price= list() 
                self.buy_order_position= list()
                self.buy_order_take_profit= list()
                self.buy_order_stop_loss= list()
                self.sell_order_datetime = list() 
                self.sell_order_price= list() 
                self.sell_order_position= list()

                # Data for 5m candlesticks
                self.data0 = self.datas[0]
                
                self.macd = bt.ind.MACD(
                    self.data0,
                    period_me1=self.p.macd1,
                    period_me2=self.p.macd2,
                    period_signal=self.p.macdsig
                )
                
                # Cross of macd and macd signal
                self.mcross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

                '''# RSI EMA
                self.rsi_ema = bt.ind.RSI_EMA(
                    self.data0,
                    period=self.p.rsi_ema
                )

                # RSI SMA
                self.rsi_sma = bt.ind.RSI_SMA(
                    self.data0,
                    period=self.p.rsi_sma
                )

                # BOLLINGER BANDS
                self.boll = bt.ind.BollingerBands(
                    self.data0.close,
                    period=self.p.boll,
                    devfactor=self.p.devfactor
                )'''

                '''# Data for 30m candlesticks
                self.data1 = self.datas[1]

                self.ema = bt.ind.EMA(
                    self.data1,
                    period=self.p.ema200
                )

                # Cross of ema200 30m candles and price
                self.ema200_30m_cross = bt.ind.CrossOver(self.data1.close, self.ema)'''
        
            def notify_order(self, order):
                """Execute when buy or sell is triggered
                Notify if order was accepted or rejected
                """
                
                if order.alive():
                    '''print("Order Alive")
                    logging.debug("Order Alive")'''

                if not order.alive() and order.ref in self.orefs:
                    self.orefs.remove(order.ref)

                if order.status == order.Completed:
                    if order.isbuy():              
                        self.holdstart = len(self) if self.dca_count == 0 else self.holdstart
                        order_side = "Buy"
                        '''print(
                            (
                                f"{order_side} Order Completed -  Size: {order.executed.size} "
                                f"@Price: {order.executed.price} "
                                f"Date: {self.data0.datetime.datetime(0)} "
                                f"Take Profit Price: {order.executed.price * (1 + (self.p.take_profit))} "
                                f"Stop Loss Price: {order.executed.price * (1- (self.p.stop_loss))} "
                                f"Value: {order.executed.value:.2f} "
                                f"Comm: {order.executed.comm:.6f} "
                                f"Order Ref. {self.orefs} "
                            )
                        )'''
                        logging.debug(
                            (
                                f"{order_side} Order Completed -  Size: {order.executed.size} "
                                f"@Price: {order.executed.price} "
                                f"Date: {self.data0.datetime.datetime(0)} "
                                f"Take Profit Price: {order.executed.price * (1 + (self.p.take_profit))} "
                                f"Stop Loss Price: {order.executed.price * (1- (self.p.stop_loss))} "
                                f"Value: {order.executed.value:.2f} "
                                f"Comm: {order.executed.comm:.6f} "
                                f"Order Ref. {self.orefs} "
                            )
                        )

                        self.buy_price_executed_order_list = [order.executed.price]
                        self.stop_loss_price_executed_order_list = [order.executed.price * (1 - (self.p.stop_loss))]
                        self.take_profit_price_executed_order_list = [order.executed.price * (1 + (self.p.take_profit))]

                        # Create lists of data
                        self.buy_order_datetime.append(self.data0.datetime.datetime(0)) 
                        self.buy_order_price.append(order.executed.price) 
                        self.buy_order_position.append(order.executed.size)
                        self.buy_order_take_profit.append(order.executed.price * (1 + (self.p.take_profit)))
                        self.buy_order_stop_loss.append(order.executed.price * (1 - (self.p.stop_loss))) 

                    else:
                        order_side = "Sell"
                        '''print(
                            (
                                f"{order_side} Order Completed -  Size: {order.executed.size} "
                                f"@Price: {order.executed.price} "
                                f"Date: {self.data0.datetime.datetime(0)} "
                                f"Value: {order.executed.value:.2f} "
                                f"Comm: {order.executed.comm:.6f} "
                                f"Order Ref. {self.orefs} "
                                f"Order Status: {order.status} - {order.getstatusname()}"
                            )
                        )'''
                        logging.debug(
                            (
                                f"{order_side} Order Completed -  Size: {order.executed.size} "
                                f"@Price: {order.executed.price} "
                                f"Date: {self.data0.datetime.datetime(0)} "
                                f"Value: {order.executed.value:.2f} "
                                f"Comm: {order.executed.comm:.6f} "
                                f"Order Ref. {self.orefs} "
                                f"Order Status: {order.status} - {order.getstatusname()}"
                            )
                        )

                        self.sell_price_executed_order_list = [order.executed.price]

                        # Create lists of data
                        self.sell_order_datetime.append(self.data0.datetime.datetime(0)) 
                        self.sell_order_price.append(order.executed.price) 
                        self.sell_order_position.append(order.executed.size)

                elif order.status in {order.Canceled, order.Margin, order.Rejected}:
                    '''print(f"{order_side} Order Canceled/Margin/Rejected")'''
                    logging.debug(f"{order_side} Order Canceled/Margin/Rejected")

                self.order = None  # indicate no order pending

            def notify_trade(self, trade):
                """Execute after each trade
                Calcuate Gross and Net Profit/loss"""
                if not trade.isclosed:

                    return

                else:
                    
                    trade_pnl_comm = [trade.pnlcomm]

                    #return trade_pnl_comm
                    
                    if trade_pnl_comm[0] > 0:

                        self.win_counter += 1
                        self.profit.append(trade_pnl_comm[0])
                        self.loss.append(0)
                        self.win_holding_time.append(len(self) - self.holdstart)
                        self.profit_sum = sum(self.profit)
                        trade_result = "Trade Won"
                        '''print(
                            f"{trade_result}"
                        )'''
                        logging.debug(
                            f"{trade_result}"
                        )
                
                    elif trade_pnl_comm[0] < 0:
                        
                        self.lost_counter += 1
                        self.loss.append(trade_pnl_comm[0])
                        self.profit.append(0)
                        self.loss_sum = sum(self.loss)
                        self.lost_holding_time.append(len(self) - self.holdstart)
                        trade_result = "Trade Lost"
                        '''print(
                            f"{trade_result}"
                        )'''
                        logging.debug(
                            f"{trade_result}"
                        )

                    self.trades_counter += 1
                    self.holding_time.append(len(self) - self.holdstart)
                    self.pnl_evol.append(trade_pnl_comm)
                    
                    #return self.win_counter, self.lost_counter, self.profit, self.loss, self.profit_sum, self.loss_sum, self.trades_counter, self.holding_time
                
                '''print(
                    f"Operational profit, Gross: {trade.pnl:.2f}, "
                    f"Net: {trade.pnlcomm:.2f}, "
                    f"Position {self.position.upopened}, "
                    f"Hold Time {len(self) - self.holdstart}, "
                    f"Trades Ended {self.trades_counter}, "
                    f"Won Trades {self.win_counter}, "
                    f"Lost Trades: {self.lost_counter}, "
                    f"Total Profit So Far: {self.profit_sum}, "
                    f"Total Loss So Far: {self.loss_sum}, "
                    f"Average Holding Time {sum(self.holding_time) / len(self.holding_time)}, "
                )'''
                logging.debug(
                    f"Operational profit, Gross: {trade.pnl:.2f}, "
                    f"Net: {trade.pnlcomm:.2f}, "
                    f"Position {self.position.upopened}, "
                    f"Hold Time {len(self) - self.holdstart}, "
                    f"Trades Ended {self.trades_counter}, "
                    f"Won Trades {self.win_counter}, "
                    f"Lost Trades: {self.lost_counter}, "
                    f"Total Profit So Far: {self.profit_sum}, "
                    f"Total Loss So Far: {self.loss_sum}, "
                    f"Average Holding Time {sum(self.holding_time) / len(self.holding_time)}, "
                )
                
                """ Calculate the actual returns """
                #self.roi = (self.broker.get_value() / self.val_start) - 1.0
                self.roi = (self.broker.get_value() / (self.val_start + sum(self.profit[:-1]) + sum(self.loss[:-1]))) - 1
                val_end = self.broker.get_value()
                
                '''print(
                    f"ROI: {100.0 * self.roi:.2f}%, Start Value {self.val_start:.2f}, "
                    f"End Value: {val_end:.2f}"
                )'''
                logging.debug(
                    f"ROI: {100.0 * self.roi:.2f}%, Start Value {self.val_start:.2f}, "
                    f"End Value: {val_end:.2f}"
                )
            
            def stop(self):
                """ Calculate the actual returns """
                self.roi_end = (self.broker.get_value() / self.val_start) - 1.0
                val_end = self.broker.get_value()
                
                '''print(
                    f"ROI: {100.0 * self.roi_end:.2f}%, Start cash {self.val_start:.2f}, "
                    f"End cash: {val_end:.2f}"
                )'''
                logging.debug(
                    f"ROI: {100.0 * self.roi_end:.2f}%, Start cash {self.val_start:.2f}, "
                    f"End cash: {val_end:.2f}"
                )

                '''print(
                    f"Total Profit {self.profit_sum}, "
                    f"Total Loss {self.loss_sum}\n"
                    f"Avg. Profit {self.profit_sum / self.win_counter}, "
                    f"Avg. Loss {self.loss_sum / self.lost_counter}\n"
                    f"Max. Profit {max(self.profit)}, "
                    f"Max. Loss {max(self.loss)}\n"
                    f"Max. Profit Deviation {(max(self.profit) / (sum(self.profit) / self.win_counter)) - 1.0}, "
                    f"Max. Loss Deviation {(max(self.loss) / (sum(self.loss) / self.lost_counter)) - 1.0}\n"
                    f"Avg. Holding Time {sum(self.holding_time) / self.trades_counter}, "
                    f"Avg. Won Trades Holding Time {sum(self.win_holding_time) / self.win_counter}, "
                    f"Avg. Lost Trades Holding Time {sum(self.lost_holding_time) / self.lost_counter}\n"
                    f"Trades Won {self.win_counter}, "
                    f"Trades Lost {self.lost_counter}\n"
                    f"Avg. Number of Trades to Compensate Loss {(self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)}, "
                    f"Trades Compensated by Loss {((self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)) * self.lost_counter * -1}\n"
                    f"Net Trades After Loss Compensation {self.win_counter - (((self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)) * self.lost_counter * -1)}"
                )'''

                logging.debug(
                    f"Backtest with Take Profit of {self.p.take_profit * 100} %\n"
                    f"Backtest with DCA 1 of {self.p.dca_price_var1 * 100} % and DCA 2 of {self.p.dca_price_var2 * 100}\n"
                    f"Total Profit {self.profit_sum}\n"
                    f"Total Loss {self.loss_sum}\n"
                    f"Avg. Profit {self.profit_sum / self.win_counter}, "
                    f"Avg. Loss {(self.loss_sum / self.lost_counter) if self.lost_counter else 0}\n"
                    f"Max. Profit {max(self.profit)}, "
                    f"Max. Loss {max(self.loss)}\n"
                    f"Max. Profit Deviation {(max(self.profit) / (sum(self.profit) / self.win_counter)) - 1.0}, "
                    f"Max. Loss Deviation {((max(self.loss) / (sum(self.loss) / self.lost_counter)) - 1.0) if self.lost_counter else 0}\n"
                    f"Avg. Holding Time {sum(self.holding_time) / self.trades_counter}, "
                    f"Avg. Won Trades Holding Time {sum(self.win_holding_time) / self.win_counter}, "
                    f"Avg. Lost Trades Holding Time {(sum(self.lost_holding_time) / self.lost_counter) if self.lost_counter else 0}\n"
                    f"Trades Won {self.win_counter}, "
                    f"Trades Lost {self.lost_counter}\n"
                    f"Avg. Number of Trades to Compensate Loss {((self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)) if self.lost_counter else 0}, "
                    f"Trades Compensated by Loss {(((self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)) * self.lost_counter * -1) if self.lost_counter else 0}\n"
                    f"Net Trades After Loss Compensation {(self.win_counter - (((self.loss_sum / self.lost_counter) / (self.profit_sum / self.win_counter)) * self.lost_counter * -1)) if self.lost_counter else 0}\n"
                    f"DCA Count: {sum(self.dca_count_list)}"
                )

                '''dict1 = {'dt':self.datetime_list, 'close':self.close_price_list, 'macd': self.macd_macd_list, 'signal': self.macd_signal_list,
                            'macd_cross': self.macd_cross_list, 'rsi_ema': self.rsi_ema_list, 'rsi_sma': self.rsi_sma_list, 'boll_bot': self.boll_bot_list,
                            'boll_mid': self.boll_mid_list, 'boll_top': self.boll_top_list}

                dict2 = {'dt': self.buy_order_datetime, 'buy_order_price': self.buy_order_price, 'buy_order_position': self.buy_order_position,
                            'buy_order_take_profit': self.buy_order_take_profit, 'buy_order_stop_loss': self.buy_order_stop_loss}

                dict3 = {'dt': self.sell_order_datetime, 'sell_order_price': self.sell_order_price, 'sell_order_position': self.sell_order_position}
                
                df1 = pd.DataFrame(dict1)
                df2 = pd.DataFrame(dict2)
                df3 = pd.DataFrame(dict3)

                #print(df1.head())
                #print(df2.head())
                #print(df3.head())

                df_final = df2.merge(df1, on='dt', how='left')
                df_final = df_final.merge(df3, on='dt', how='outer')
                
                #print(df_final.head())

                # writing to Excel 
                df_bt_excel = pd.ExcelWriter('freqtrade/finamom/Futures/report/'+coin_name+'_report.xlsx') 
        
                # write DataFrame to excel 
                df_final.to_excel(df_bt_excel) 
        
                # save the excel 
                df_bt_excel.save()'''
            
            def next(self):

                '''self.datetime_list.append(self.data0.datetime.datetime(0))
                self.close_price_list.append(self.data0.close[0])
                self.macd_macd_list.append(self.macd.macd[0])
                self.macd_signal_list.append(self.macd.signal[0])
                self.macd_cross_list.append(self.mcross[0])
                self.rsi_ema_list.append(self.rsi_ema[0])
                self.rsi_sma_list.append(self.rsi_sma[0])
                self.boll_mid_list.append(self.boll.lines.mid[0])
                self.boll_bot_list.append(self.boll.lines.bot[0])
                self.boll_top_list.append(self.boll.lines.top[0])'''
                
                #close_price_vs_ema200 = "Close Price > EMA 200" if self.data0.close[0] > self.ema[0] else "Close Price <= EMA 200"

                #if self.order:
                if self.orefs:
                    return  # pending orders do nothing

                '''print(
                    f"DateTime {self.data0.datetime.datetime(0)}, "
                    f"Close 5m: {self.data0[0]:.2f}, "# - {close_price_vs_ema200}, "
                    f"MACDcross {self.mcross[0]}, "
                    f"Get Position {self.getposition(self.data0).size}, "
                    f"Position Size {self.position.size}"
                )
                logging.debug(
                    f"DateTime {self.data0.datetime.datetime(0)}, "
                    f"Close 5m: {self.data0[0]:.2f}, "# - {close_price_vs_ema200}, "
                    f"MACDcross {self.mcross[0]}, "
                    f"Get Position {self.getposition(self.data0).size}, "
                    f"Position Size {self.position.size}"
                )'''

                if not self.position:
                    
                    if (self.mcross[0] > 0.0 and self.macd[0] < 0.0): # and (self.data0.close[0] > self.ema[0]): # conditions for entering the market
                        
                        '''print("Time to Buy.")
                        logging.debug("Time to Buy.")'''

                        # Data needed for the self.buy()
                        self.buy_price = self.data0.close[0] # price of the 5m candle after triggering singal

                        self.long = self.buy(
                            price=self.buy_price,
                            size=self.broker.get_cash() / self.data0.close * self.p.portfolio_frac,
                            exectype=bt.Order.Limit
                            )

                        # Data needed for self.sell() and dca position
                        self.order = self.long                
                        
                        # Tuples needed to reference values to another functions and instances
                        self.orefs = [self.order.ref] # order reference number

                        '''print(
                                f"{self.orefs}, "
                                f"Position Price = {self.order.price}, "
                                f"Long Size = {self.long.price}, "
                                f"Long Size = {self.long.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac}, "
                            )

                        logging.debug(
                                f"{self.orefs}, "
                                f"Position Price = {self.order.price}, "
                                f"Long Price = {self.long.price}, "
                                f"Long Size = {self.long.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac}, "
                            )'''

                        self.dca_count = 0 if self.dca_count == '' else self.dca_count
                        self.dca_count_list.append(self.dca_count)

                elif self.position:  # in the market
                    
                    # sell with take_profit or stop_loss:
                    if self.data0.close[0] <= (self.position.price * (1 - self.p.stop_loss)):
                        
                        '''print("You Should Sell for Loss.")
                        logging.debug("You Should Sell for Loss.")'''
                        self.order = self.sell(
                            price=(self.position.price * (1 - self.p.stop_loss)),
                            size=self.getposition(self.data0).size,
                            exectype=bt.Order.Limit
                            )

                        self.orefs = [self.order.ref]

                        '''print(f"{self.orefs}, "
                                f"Stop Loss Position Price = {self.position.price * (1 - self.p.stop_loss)}, "
                                f"Close = {self.data0.close[0]}, "
                                f"DateTime {self.data0.datetime.datetime(0)}, "
                                f"Position Price = {self.position.price}, "
                                f"Position Size = {self.position.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac} "
                            )

                        logging.debug(f"{self.orefs}, "
                                f"Stop Loss Position Price = {self.position.price * (1 - self.p.stop_loss)}, "
                                f"Close = {self.data0.close[0]}, "
                                f"DateTime {self.data0.datetime.datetime(0)}, "
                                f"Position Price = {self.position.price}, "
                                f"Position Size = {self.position.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac} "
                            )'''

                        self.dca_count = 0
                        
                        self.dca_count_list.append(self.dca_count)

                    elif self.data0.close[0] >= self.position.price * (1 + self.p.take_profit):
                        
                        '''print("You Should Sell for Profit.")
                        logging.debug("You Should Sell for Profit.")'''
                        self.order = self.sell(
                            price=(self.position.price * (1 + self.p.take_profit)),
                            size=self.getposition(self.data0).size,
                            exectype=bt.Order.Limit
                            )

                        self.orefs = [self.order.ref]

                        '''print(f"{self.orefs}, "
                                f"Take Profit Price = {self.position.price * (1 + self.p.take_profit)}, "
                                f"Close = {self.data0.close[0]}, "
                                f"DateTime {self.data0.datetime.datetime(0)}, "
                                f"Position Price = {self.position.price}, "
                                f"Position Size = {self.position.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac} "
                            )

                        logging.debug(f"{self.orefs}, "
                                f"Take Profit Price = {self.position.price * (1 + self.p.take_profit)}, "
                                f"Close = {self.data0.close[0]}, "
                                f"DateTime {self.data0.datetime.datetime(0)}, "
                                f"Position Price = {self.position.price}, "
                                f"Position Size = {self.position.size}, "
                                f"Cash = {self.broker.get_cash()}, "
                                f"Percentage Invested = {self.p.portfolio_frac} "
                            )'''

                        self.dca_count = 0

                        self.dca_count_list.append(self.dca_count)
            
                    # if the difference between the start of the backtest and the trade notification is bigger than the hold limit parameter
                    #elif (len(self) - self.holdstart) >= self.p.hold:

                    #    pass
                    
                    else:
                        # If there is a limit to the holding time this condition is true if holding time < holding limit time
                        
                        '''print("Still in the Market Condition: "+str(len(self))+" - "+str(self.holdstart)+" = "+str(len(self) - self.holdstart))#+" : "+str(self.p.hold)+"")
                        logging.debug("Still in the Market Condition: "+str(len(self))+" - "+str(self.holdstart)+" = "+str(len(self) - self.holdstart))#+" : "+str(self.p.hold)+"")'''

                        if self.position.size > 0:
                        
                            '''print(f"Position Size > 0 ---> {self.position.size}")
                            logging.debug(f"Position Size > 0 ---> {self.position.size}")'''

                            if self.data0.close[0] > self.position.price * (1 - self.p.dca_price_var1) and self.dca_count == 0:
                                
                                '''print("No DCA Yet")
                                logging.debug("No DCA Yet")'''

                            elif self.data0.close[0] <= self.position.price * (1 - self.p.dca_price_var1) and self.dca_count == 0:
                                
                                self.buy_price = self.data0.close[0] # price of the 5m candle after triggering signal 
                                
                                self.long = self.buy(
                                    price=self.position.price * (1 - self.p.dca_price_var1),
                                    size=self.getposition(self.data0).size,
                                    exectype=bt.Order.Limit,
                                    )

                                # Data needed for self.sell() and dca position
                                self.order = self.long 

                                # Tuples needed to reference values to another functions and instances
                                self.orefs = [self.order.ref] # order reference number

                                # DCA counter == 1
                                self.dca_count = 1
                                self.dca_count_list.append(self.dca_count)

                                '''print("DCA 1 About to be Executed")
                                print(
                                    f"Position.Size {self.position.size}, "
                                    f"GetPosition {self.getposition(self.data0).size}, "
                                    f"Position.Price {self.position.price}"
                                    )
                                logging.debug("DCA 1 About to be Executed")
                                logging.debug(
                                            f"Position.Size {self.position.size}, "
                                            f"GetPosition {self.getposition(self.data0).size}, "
                                            f"Position.Price {self.position.price}"
                                            )'''

                            elif self.data0.close[0] <= self.position.price * (1 - self.p.dca_price_var2) and self.dca_count == 1:
                                
                                self.buy_price = self.data0.close[0] # price of the 5m candle after triggering signal 
                                
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
                                self.dca_count_list.append(self.dca_count)

                                '''print("DCA 2 About to be Executed")
                                print(
                                    f"Position.Size {self.position.size}, "
                                    f"GetPosition {self.getposition(self.data0).size}, "
                                    f"Position.Price {self.position.price}"
                                    )
                                logging.debug("DCA 2 About to be Executed")
                                logging.debug(
                                            f"Position.Size {self.position.size}, "
                                            f"GetPosition {self.getposition(self.data0).size}, "
                                            f"Position.Price {self.position.price}"
                                            )'''

                            elif self.data0.close[0] <= self.position.price * (1 - self.p.dca_price_var3) and self.dca_count == 2:
                                
                                self.buy_price = self.data0.close[0] # price of the 5m candle after triggering signal 
                                
                                self.long = self.buy(
                                    price=self.position.price * (1 - self.p.dca_price_var3),
                                    size=self.getposition(self.data0).size,
                                    exectype=bt.Order.Limit,
                                    )

                                # Data needed for self.sell() and dca position
                                self.order = self.long 

                                # Tuples needed to reference values to another functions and instances
                                self.orefs = [self.order.ref] # order reference number

                                # DCA counter == 2
                                self.dca_count = 3
                                self.dca_count_list.append(self.dca_count)

                                '''print("DCA 2 About to be Executed")
                                print(
                                    f"Position.Size {self.position.size}, "
                                    f"GetPosition {self.getposition(self.data0).size}, "
                                    f"Position.Price {self.position.price}"
                                    )
                                logging.debug("DCA 2 About to be Executed")
                                logging.debug(
                                            f"Position.Size {self.position.size}, "
                                            f"GetPosition {self.getposition(self.data0).size}, "
                                            f"Position.Price {self.position.price}"
                                            )'''
                            
                            else:

                                '''print("DCA 1 Used Already") if self.dca_count == 1 else print("DCA 1 and 2 Used Already")
                                print(
                                    f"Get Position {self.getposition(self.data0).size}, "
                                    f"Price {self.position.price}, "
                                    f"Stop Loss Price {self.position.price * (1 - self.p.stop_loss)}, "
                                    f"Take Profit Price {self.position.price * (1 + self.p.take_profit)}"
                                    )
                                logging.debug("DCA 1 Used Already") if self.dca_count == 1 else print("DCA 1 and 2 Used Already")
                                logging.debug(
                                    f"Get Position {self.getposition(self.data0).size}, "
                                    f"Price {self.position.price}, "
                                    f"Stop Loss Price {self.position.price * (1 - self.p.stop_loss)}, "
                                    f"Take Profit Price {self.position.price * (1 + self.p.take_profit)}"
                                    )'''
                                
                                pass

                else:

                    '''print("WHAT")
                    logging.debug("WHAT")'''

        #cerebro = bt.Cerebro(cheat_on_open=True)
        cerebro = bt.Cerebro()

        # Amount of starting cash
        cerebro.broker.set_cash(1000)

        # First Dataframe (5min candles)
        data5m = bt.feeds.GenericCSVData(
            dataname='freqtrade/finamom/Futures/datas/'+coin_name+'_futures_5m.csv',
            dtformat="%Y-%m-%d %H:%M:%S",
            timeframe=bt.TimeFrame.Ticks,
            compression=5,
            #fromdate=datetime.datetime(2020, 9, 1, 0, 0, 0),
            #todate=datetime.datetime(2020, 11, 15, 0, 0, 0),
            datetime=1,
            high=3,
            low=4,
            open=2,
            close=5,
            volume=6,
            openinterest=-1,
        )

        #cerebro.resampledata(data5m, timeframe=bt.TimeFrame.Minutes, compression=5)

        '''# Second Dataframe (30min canldes)
        data30m = bt.feeds.GenericCSVData(
            dataname='freqtrade/finamom/Futures/datas/zil_futures_30m.csv',
            dtformat="%Y-%m-%d %H:%M:%S",
            timeframe=bt.TimeFrame.Ticks,
            compression=30,
            fromdate=datetime.datetime(2020, 10, 2, 0, 0, 0),
            #todate=datetime.datetime(2020, 11, 15, 0, 0),
            datetime=1,
            high=2,
            low=3,
            open=4,
            close=5,
            volume=6,
            openinterest=-1,
        )'''

        #cerebro.resampledata(data30m, timeframe=bt.TimeFrame.Minutes, compression=30)

        cerebro.adddata(data5m, name=coin_name)
        #cerebro.adddata(data30m, name='SNXUSDT 30M')

        # Add strategy
        cerebro.addstrategy(MACD)

        # Add comission (BitmexComissionInfo)
        #cerebro.broker.addcommissioninfo(BitmexComissionInfo())
        #cerebro.broker = bt.brokers.BackBroker(slip_fixed=0.0)
        cerebro.broker.setcommission(commission=0.0)

        # Add TimeReturn Analyzers to benchmark data
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn, _name="alltime_roi", timeframe=bt.TimeFrame.NoTimeFrame
        )

        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            data=data5m,
            _name="benchmark",
            timeframe=bt.TimeFrame.NoTimeFrame,
        )

        # Execute
        results = cerebro.run()
        st0 = results[0]

        #logging.debug(results)
        #logging.debug(st0)

        #alyzers_list=[]
        #results_list=[]

        #for alyzer in st0.analyzers:
        #    alyzer.print()
        #    alyzers_list.append(alyzer)

        #print(alyzers_list)
        #logging.debug(alyzers_list)

        #cerebro.plot(iplot=False, style="bar")
