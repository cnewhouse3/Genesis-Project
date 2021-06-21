# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:44:07 2021

@author: mouss
"""

from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import statistics as stats
import pandas as pd
import numpy as np
import math
import numpy as np
from collections import defaultdict
from scipy.signal import lfilter
from IPython.core.display import display, HTML
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import matplotlib
import seaborn as sns



from plotly.subplots import make_subplots
import plotly.graph_objects as go

from numpy import log, sqrt, std, subtract, cumsum, polyfit
def hurst1(ts):
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]  # This line throws the Error

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
class backtest:
    def __init__(self, price, resid_lst, date_lst, sig_size, start, close, close_offset, cost, title):
        #data
        self.title = title
        self.buy_start = start[0]
        self.sell_start = start[1]
        self.price = price
        self.resid_lst = resid_lst
        self.date_lst = date_lst
        self.state = 0
        #portfolio and trades
        self.sig_size = sig_size
        #trades
        self.open_trades = []
        self.closed_trades_B = []
        self.closed_trades_S = []
        self.open_hedges = []
        self.closed_hedges = []
        self.cur_ratios = []
        #pnl
        self.pnl = []
        self.realized = []

        #historical pos
        self.hpos_px = []
        self.hpos_sz = []
        #state
        self.long = 0
        self.short = 0
        self.sz = []
        #exit positions
        self.cost = cost
        self.long_close = close[0]
        self.short_close = close[1]
        self.close_offset = close_offset

    #return MTM pnl, tr, and hedged
    def MTM(self, px):
        pnl = 0
        for t in self.open_trades:
            tpnl = t.MTM(px)
            pnl += tpnl
        return pnl
   #update data when no trade
    def noTrade(self, date, sig, px):
        self.closeT(date, sig, px)

            #self.checkRealized()  
    def checkRealized(self):
        if len(self.realized) == 0:
            self.realized.append(0)
        else:
            self.realized.append(self.realized[-1])
    def closeT(self, date, sig, px):
        #if short close position
        trade_pnl = 0
        tr_pnl = 0
        trades = []
        val = self.close_offset
        if val !=0:
            while self.open_trades:
                t = self.open_trades.pop()
                ##CHANGED THIS
                if (t.signal > 0 and (t.signal*val >= sig or sig <= self.long_close)) or (t.signal < 0 and (t.signal*val<= sig or sig >= self.short_close)):
                    t.close(date, sig, px)
                    trade_pnl+= t.final_pnl
                    if self.short == 1:
                        self.closed_trades_S.append(t)
                    else:
                        self.closed_trades_B.append(t)
                else:
                    trades.append(t)
        else:
            while self.open_trades:
                t = self.open_trades.pop()
                if (t.signal > 0 and  sig <= self.long_close) or (t.signal < 0 and sig >= self.short_close):
                    t.close(date, sig, px)
                    trade_pnl+= t.final_pnl
                    if self.short == 1:
                        self.closed_trades_S.append(t)
                    else:
                        self.closed_trades_B.append(t)
                else:
                    trades.append(t)
        self.open_trades = trades
        if len(self.open_trades) == 0:
            self.long = 0 
            self.short = 0
            if len(self.realized) >0:
                self.realized.append(self.realized[-1]+ trade_pnl)
            else:
                self.realized.append(trade_pnl)
            self.pnl.append(0) 
            self.state = 0
            self.sz.append(0)
        else:
            if len(self.realized) >0:
                self.realized.append(self.realized[-1]+ trade_pnl)
            else:
                self.realized.append(trade_pnl)
            self.pnl.append(self.MTM(px))
            self.state = self.stateMax()
            self.sz.append(self.size())
        return trade_pnl
    def size(self):
        sz = 0
        for x in self.open_trades:
            sz+= x.sz
        return sz
    def stateMax(self):
        state = []
        for x in self.open_trades:
            state.append(x.signal)
        if stats.mean(state) > 0:
            return max(state)
        else:
            return min(state)
#nested classes for trades/hedges
class trade:
    def __init__(self, bs, date, sig, px, sz, cost):
        self.bs = bs
        self.start_date = date
        self.signal = sig
        
        if self.bs == 1:
            initial_cost = cost*abs(sz)/100*-1
            self.px = (initial_cost*100/abs(sz)-px)*-1
            #self.px = px
        else:
            initial_cost = cost*abs(sz)/100*-1
            self.px = (initial_cost*100/abs(sz)+px)

        self.sz = sz
        self.end_date = float('nan')
        self.close_px = float('nan')
        self.close_sig = float('nan')
        self.open = True
        self.final_pnl = 0
        self.cost = cost
    def close(self, date, sig, px):
        self.end_date = date
        self.close_sig = sig
        self.close_px = px
        self.close_sig = sig
        self.open = False
        self.final_pnl += self.MTM(px)
        self.final_pnl -= self.cost*abs(self.sz)/100
    def MTM(self, px):
        pnl = 0
        pnl = (self.px - px)*abs(self.sz)/100
        if self.bs==1:
            pnl = (px-self.px)*abs(self.sz)/100
        return pnl
def drawDown(px, date_lst):
    draw_down = []
    dates = []
    close_to_zero = []
    dates_ctz = []
    for x in range(0, len(px)):
        if x+1 <= len(px)-1:
            min_val = min(px[x+1:])
            draw_down.append([min_val-px[x]])
            index = px.index(min_val)
            dates.append([pd.to_datetime(date_lst[x]).strftime("%Y-%m-%d"), pd.to_datetime(date_lst[index]).strftime("%Y-%m-%d")])

        temp_date = date_lst[x]
        if x +1 < len(px)-1:
            z = x+1
            while z< len(px)-1 and  px[x] > px[z]:
                z+=1
            temp_date = date_lst[z]

            
        #close_to_zero.append(min(temp))
        #print(temp)

        dates_ctz.append((pd.to_datetime(temp_date)-pd.to_datetime(date_lst[x])).days)
        close_to_zero.append([pd.to_datetime(date_lst[x]).strftime("%Y-%m-%d"),pd.to_datetime(temp_date).strftime("%Y-%m-%d")])

    return min(draw_down), dates[draw_down.index(min(draw_down))], max(dates_ctz), close_to_zero[dates_ctz.index(max(dates_ctz))]
        
            
            
            
def runBackTest_Binary(test, rl, rs):
    state_lst = []
    for index, row in enumerate(test.price):

        rx = test.resid_lst[index]
        px = test.price[index]
        #get leg px
#        if test.date_lst[index].day == 28 and test.date_lst[index].month==1:
#            print(test.state)
#            print(test.date_lst[index])
#            break
        rx = int(rx)
        if rx in test.sig_size:
            #get leg and hedge siziing
            size_t = test.sig_size[rx]
            
            if size_t > 0:
                #no position

                if (test.long == 0) and (test.short==0) and (rx >= test.buy_start) and rl:

                    sz = test.sig_size[rx]
#                    for x in range(int(test.buy_start), int(rx+1)):
#                        temp_size = (test.sig_size[x]-test.sig_size[x-1])
#                        t1 = trade(1,test.date_lst[index], x, px, temp_size, test.cost[index])
#                        #add trade/hedge might need to optimize later
#                        test.open_trades.append(t1)
#                        sz += temp_size
                    t1 = trade(1,test.date_lst[index], rx, px,sz, test.cost[index])
                    test.open_trades.append(t1)   

                    test.long = 1
                    test.state = rx
                    #record pnl data
                    pnl= test.MTM(px)
                    test.pnl.append(pnl)
                    test.state = rx
                    #upadte realized
                    test.checkRealized()
                    if len(test.sz) >0:
                        test.sz.append(sz)
                    else:
                        test.sz.append(sz)
                #closing short/reverse long
                elif (test.short==1) and rx >= test.short_close:

                    test.closeT(test.date_lst[index], rx, px)
                    test.long = 0
                    test.short = 0
                    test.state = 0
                    if len(test.open_trades) ==0:
                        test.long = 0
                        test.short = 0
                        test.state = 0
                        #reverse
                        if rx >= test.buy_start and rl:
                            sz=test.sig_size[rx]
#                            for x in range(int(test.buy_start), int(rx+1)):
#                                temp_size = test.sig_size[x]-test.sig_size[x-1]
#                                t1 = trade(1,test.date_lst[index], x, px, temp_size, test.cost[index])
#                                #add trade/hedge might need to optimize later
#                                test.open_trades.append(t1)
#                                sz += temp_size
                            t1 = trade(1,test.date_lst[index], rx, px,sz, test.cost[index])
                            test.open_trades.append(t1)
                            test.sz[-1]=sz
                            test.state = rx
                            test.long=1
                            #record pnl data
                            test.pnl[-1]=(test.MTM(px))

                #addPosition if long already
                elif (test.long==1) and (rx >test.state) and rl:

                    sz = 0

                    for x in range(int(test.state+1), int(rx+1)):
                        temp_size = test.sig_size[x]-test.sig_size[x-1]
                        t1 = trade(1,test.date_lst[index], x, px, temp_size, test.cost[index])
                        #add trade/hedge might need to optimize later
                        test.open_trades.append(t1)
                        sz += temp_size   
                    test.long = 1
                    test.state = rx
                    #upadte current sizing
                    pnl= test.MTM(px)
                    test.pnl.append(pnl)
                    test.state = rx
                    test.checkRealized()
                    test.sz.append(sz+test.sz[-1])
  
                else:
                    test.closeT(test.date_lst[index], rx, px)
#                print(sz, test.state, state_lst[-1], rx)
#                print(test.date_lst[index])
#                if rx==4:
#                    break
            #Short Event
            elif size_t < 0:


                #no position, go short
                if (test.long == 0) and (test.short==0) and (rx  <= test.sell_start) and rs:

                    sz = test.sig_size[rx]
#                    for x in range(int(test.sell_start*-1), int((rx)*-1+1)):
#                        temp_size = test.sig_size[x*-1]-test.sig_size[(x-1)*-1]
#                        t1 = trade(0,test.date_lst[index], x*-1, px, temp_size, test.cost[index])
#                        #add trade/hedge might need to optimize later
#                        test.open_trades.append(t1)
#                        sz += temp_size

                    t1 = trade(0,test.date_lst[index],rx, px, sz, test.cost[index])
                    #add trade/hedge might need to optimize later
                    test.open_trades.append(t1)
                    test.short = 1
                    test.state = rx
                    #record pnl data
                    pnl= test.MTM(px)
                    test.pnl.append(pnl)
                    test.state = rx
                    #upadte realized
                    
                    test.checkRealized()
                    if len(test.sz) >0:
                        test.sz.append(sz)
                    else:
                        test.sz.append(sz)

                #closing long/reverse short
                elif (test.long==1) and rx <= test.long_close:
                    test.closeT(test.date_lst[index], rx, px)
                    test.long = 0
                    test.short = 0
                    test.state = 0
                    if len(test.open_trades) ==0:
                        test.long = 0
                        test.short = 0
                        test.state = 0
                        #reverse
                        if rx <= test.sell_start and rs:
                            sz = test.sig_size[rx]
#                            for x in range(int(test.sell_start*-1), int(rx*-1+1)):
#                                temp_size = test.sig_size[x*-1]-test.sig_size[(x-1)*-1]
#                                t1 = trade(0,test.date_lst[index], x*-1, px, temp_size, test.cost[index])
#                                #add trade/hedge might need to optimize later
#                                test.open_trades.append(t1)
#                                sz += temp_size
                            t1 = trade(0,test.date_lst[index],rx, px, sz, test.cost[index])
                            #add trade/hedge might need to optimize later
                            test.open_trades.append(t1)
                            test.sz[-1]+=sz
                            test.short = 1
                            test.state = rx
                            test.pnl[-1]=(test.MTM(px))
                            

                #addPosition if short already
                elif (test.short==1) and (rx < test.state) and (rx <= test.sell_start) and rs:
                    sz = 0
                    for x in range(int(test.state*-1+1),  int(rx*-1+1)):
                        temp_size = test.sig_size[x*-1]-test.sig_size[(x-1)*-1]
                        t1 = trade(0,test.date_lst[index], x*-1, px, temp_size, test.cost[index])
                        #add trade/hedge might need to optimize later
                        test.open_trades.append(t1)
                        sz+= temp_size
                    #upadte current sizing
                    pnl= test.MTM(px)
                    test.pnl.append(pnl)
                    test.state = rx
                    test.checkRealized()
                    test.sz.append(sz+test.sz[-1])
                else:
                    test.closeT(test.date_lst[index], rx, px)
            else:
                test.noTrade(test.date_lst[index], rx, px)
        elif len(test.open_trades) >0:

            test.closeT(test.date_lst[index], rx, px)
        else: 

            test.noTrade(test.date_lst[index], rx, px)
        state_lst.append(test.state)
    inputs = [test.sig_size, test.buy_start, test.sell_start, test.long_close, test.short_close, test.close_offset, test.cost[index]]
    input_index = ['Sizing', 'Buy Start', 'Sell Start', 'Buy Close', 'Sell Close', 'Close Offset', '1-Way Cost']
    inputs = pd.Series(inputs)
    inputs.index = input_index
    print(inputs)
    
    #print(len(test.date_lst), print(len(test.pnl)), print(len(test.sz)))
    pnl1 = []
    for x in range(len(test.realized)):
        pnl = test.realized[x]+test.pnl[x]
        pnl1.append(pnl)
#######################

    trade_df = []
    for x in test.closed_trades_B:
        date_diff = int((x.end_date-x.start_date).days)
        trade_df.append([x.bs, x.start_date, x.signal, x.sz, x.end_date, x.px, x.close_px, x.close_sig, x.final_pnl, x.cost, date_diff])
    for x in test.closed_trades_S:
        date_diff = int((x.end_date-x.start_date).days)
        trade_df.append([x.bs, x.start_date, x.signal, x.sz, x.end_date,x.px,  x.close_px, x.close_sig, x.final_pnl, x.cost, date_diff])
    trade_df = pd.DataFrame(trade_df)
    trade_df.columns = ['Buy/Sell', 'Start Date', 'Sig', 'Size', 'End Date','Open Price', 'Close Px', 'Close Sig', 'Final PNL', 'Cost', 'Holding Period']
############
    buy_ret = []
    pos_buy = 0
    buy_sz = 0 
    buy_sig = 0
    buy_dates = []
    buy_dict = defaultdict(int)
    buy_sigD = defaultdict(int)
    
    for x in test.closed_trades_B:
        buy_dict[x.start_date] += x.final_pnl
        buy_sigD[x.start_date] = x.signal
        buy_ret.append(x.final_pnl)
        buy_sz += x.sz
        if x.final_pnl > 0:
            pos_buy+=1
        if x.start_date not in buy_dates:
            buy_dates.append(x.start_date)
            buy_sig +=1
    pos_buy = 0
    for k in buy_dict:
        if buy_dict[k] > 0:
            pos_buy+=1
    
    sell_ret = []
    pos_sell = 0
    sell_sz = 0
    sell_sig = 0
    sell_dates = []
    sell_dict = defaultdict(int)
    sell_sigD = defaultdict(int)
    for x in test.closed_trades_S:
        sell_dict[x.start_date] += x.final_pnl
        sell_sigD[x.start_date] = x.signal
        sell_ret.append(x.final_pnl)
        sell_sz += x.sz
        if x.final_pnl > 0:
            pos_sell+=1
        if x.start_date not in sell_dates:
            sell_dates.append(x.start_date)
            sell_sig +=1  
    pos_sell = 0
    for k in sell_dict:
        if sell_dict[k] > 0:
            pos_sell+=1
    try:
        b_mean = sum(buy_ret)/buy_sz*100
        b_pnl = sum(buy_ret)
        b_acc = pos_buy/len(buy_dict)
        b_numb = buy_sig
        b_sz = buy_sz
    except:
        b_mean = "Broken"
        b_pnl = "Broken"
        b_acc = "Broken"
        b_numb = "Broken"
        b_sz = "Broken"   
    try:
        s_mean = sum(sell_ret)/sell_sz*100*-1
        s_pnl = sum(sell_ret)
        s_acc = pos_sell/len(sell_dict)
        s_numb = sell_sig
        s_sz = sell_sz
    except:
        s_mean = "Broken"
        s_pnl = "Broken"
        s_acc = "Broken"
        s_numb = "Broken"
        s_sz = "Broken"
    try:
        all_trades = buy_ret + sell_ret
        total_pnl = pd.DataFrame(pnl1)
        change = total_pnl.diff(1)
        change = change.dropna()
        change.columns = ['Values']
        mean_1d_chg = stats.mean(change['Values'] * 250)
        std_1d_chg = stats.stdev(change['Values']) * (250) ** (.5)
        sharpe = stats.mean(change['Values']) * 250 / (stats.stdev(change['Values']) * (250) ** (.5))
        final_pnl = pnl1[-1]
    except:
        mean_1d_chg = "Broken"
        std_1d_chg = "Broken"
        sharpe = "Broken"
        final_pnl = "Broken"
    #final_pnl, mean_1d_chg, sharpe, std_1d_chg = method_name(buy_ret, pnl1, sell_ret)
    #try:
    draw_down, date_dd, num_days_zero, zero_period = drawDown(pnl1, test.date_lst)
#    except:
#        draw_down = ["Broken"]
#        date_dd = "Broken"
#        num_days_zero = "Broken"
#        zero_period = "Broken"
    #print(draw_down, date_dd)
    data = [ b_mean,b_pnl, b_acc, b_numb, b_sz,s_mean,s_pnl, s_acc, s_numb, s_sz, mean_1d_chg, std_1d_chg, sharpe, draw_down[0], date_dd, num_days_zero, zero_period,  final_pnl  ]
    
    
    index = [ "Buy Mean", "Buy PNL", "Buy Accuracy", "Buy Number", "Buy Total Size", "Sell Mean", "Sell PNL", "Sell Accuracy", "Sell Number", "Sell Total Size", "Annualized Mean", "Annualized Std", "Sharpe", 'Max Drawdown', 'Max Draw Down Period', 'Max Days for 0 PNL', 'Zero Period',  "Total PNL"]
    data = pd.DataFrame(data)
    data.index = index
    data = data
    data.columns = ['Values']
    #(data.style.set_table_styles(styles))
    #data = data.T
    df = pd.DataFrame([test.price, test.resid_lst, test.sz, state_lst, pnl1, test.cost]).T
    df.columns = ['px', 'sig', 'size', 'state', 'PNL', 'cost']
    df.index = pd.to_datetime(test.date_lst)
    df["Date"] = df.index
    print(data)
    trades_pnl_vals = []
    trades_sig_vals = []
    
    for x in buy_dict:
        trades_pnl_vals.append(buy_dict[x])
        trades_sig_vals.append(buy_sigD[x])
    for x in sell_dict:
        trades_pnl_vals.append(sell_dict[x])
        trades_sig_vals.append(sell_sigD[x])  
    df1 = df
    df['PNL CHG'] = df['PNL'].diff(1)

    window = 100
    df = df.dropna()
    df =  df.reset_index(drop=True)

 
    window =min(100, int(round(df.shape[0]/10, 0)))
    for index, row in df.iterrows():
        if index > window:
            #print(data_out.iloc[index-100: index]['PNL CHG'])
            mean = df.iloc[index-window: index]['PNL CHG'].mean()
            std = stats.stdev(df.iloc[index-window: index]['PNL CHG'])
            if std == 0:
                last_sharpe = df1.iloc[index-1]['SHARPE']
                df.at[index, "SHARPE"] = last_sharpe
            else:
                df.at[index, "SHARPE"] = mean * 255 / (std* (255) ** (.5))
        elif index > 2:
            mean = df.iloc[0: index]['PNL CHG'].mean()
            std = stats.stdev(df.iloc[0: index]['PNL CHG'])
            if std ==0:
                df.at[index, "SHARPE"] = 0
            else:
                df.at[index, "SHARPE"] = mean * 255 / (std* (255) ** (.5))
        else:
            df.at[index, "SHARPE"] = 1
    #look_back = max(-200, df.shape[0]*-1)
    look_back = max(-200, df.shape[0]*-1)
    date_start = df.iloc[look_back]["Date"]
    date_end = df.iloc[-1]['Date']

    ################
    fig = make_subplots(rows=3, cols=2, subplot_titles=('PNL', 'Price','Signal', 'Positions', 'Signals', 'Rolling Sharpe'),horizontal_spacing = 0.05,vertical_spacing = 0.02,)
    
    #################
    name1 = 'PNL'
    name2 = 'px'

    r1 = 1
    fig.add_trace(go.Scatter(x=df['Date'], y=df[name1]),
                  row=r1, col=1)
    fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=1)
    fig.update_yaxes( range=[min(df.iloc[look_back:][name1]), max(df.iloc[look_back:][ name1])],  row=r1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], y=df[name2]),
                  row=r1, col=2)
    fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=2)
    fig.update_yaxes( range=[min(df.iloc[look_back:][name2]), max(df.iloc[look_back:][name2])],  row=r1, col=2)
    ################
    #################
    name1 = 'sig'
    name2 = 'size'
    r1 =2
    import plotly.express as px
    fig.add_trace(go.Scatter(x=df['Date'], y=df[name1]),
                  row=r1, col=1)
    fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=1)
    fig.update_yaxes( range=[min(df.iloc[look_back:][name1]), max(df.iloc[look_back:][ name1])],  row=r1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], y=df[name2]),
                  row=r1, col=2)
    fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=2)
    fig.update_yaxes( range=[min(df.iloc[look_back:][name2]), max(df.iloc[look_back:][name2])],  row=r1, col=2)
    ################
    #################
    name1 = 'Signals'
    name2 = 'SHARPE'
    r1 =3  
    fig.add_trace(go.Scatter(x=trades_sig_vals, y=trades_pnl_vals,  mode='markers'),
                  row=r1, col=1)
    #fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=1)
    #fig.update_yaxes( range=[min(df.iloc[look_back:][name1]), max(df.iloc[look_back:][ name1])],  row=r1, col=1)
    plt.show()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[name2]),
                  row=r1, col=2)
    fig.update_xaxes( type="date", range=[date_start,date_end], row=r1, col=2)
    fig.update_yaxes( range=[min(df.iloc[look_back:][name2]), max(df.iloc[look_back:][name2])],  row=r1, col=2)
    ################
    
    fig.update_layout(height=3000, width=2100,
                  title_text="BACKTEST_DATA")
   
    import plotly.io as pio
    #pio.renderers.default = 'svg'
    pio.renderers.default = 'browser'
    fig.show()
    return df, data, trade_df, fig

def method_name(buy_ret, pnl1, sell_ret):
    try:
        all_trades = buy_ret + sell_ret
        total_pnl = pd.DataFrame(pnl1)
        change = total_pnl.diff(1)
        change = change.dropna()
        change.columns = ['Values']
        mean_1d_chg = stats.mean(change['Values'] * 250)
        std_1d_chg = stats.stdev(change['Values']) * (250) ** (.5)
        sharpe = stats.mean(change['Values']) * 250 / (stats.stdev(change['Values']) * (250) ** (.5))
        final_pnl = pnl1[-1]
    except:
        mean_1d_chg = "Broken"
        std_1d_chg = "Broken"
        sharpe = "Broken"
        final_pnl = "Broken"
    return final_pnl, mean_1d_chg, sharpe, std_1d_chg


#%% Example Back Test
import os
cwd = os.getcwd()
data =  pd.read_csv(str(cwd)+'\\data\\'+"backtest_dataframe.csv")
data['date'] = pd.to_datetime(data['date'])
print(data.columns)
data= data[['date', 'z_scores_btc_funding_rates', 'btc_price_usd_close']]
data.dropna(inplace=True)
data = data[data['z_scores_btc_funding_rates'] != 0]
#reverse direction of data
data = data[::-1]
data.reset_index(drop=True, inplace=True)

data['cost'] = 0
size = {-5:-500, -4:-400, -3:-300, -2:-200, -1.5:-150, -1:-100, 0:0, 1:100, 1.5:150, 2:200, 3:300, 4:400, 5:500}
test = backtest(list(data['btc_price_usd_close']), list(data['z_scores_btc_funding_rates']), list(data['date']),size, [1,2], [-2, -1], 0, list(data['cost']),  'z_scores_btc_funding_rates')
'''
backtest(price, resid_lst, date_lst, sig_size, start, close, close_offset, cost, title)
price = list(price data frame column)
resid_lst = list(of signal z score column)
date_lst = list(of dates)
sig_size = dictionary of size to trade for each signal. Will round to nearest sig size number
start = list where first index is the signal level you will start buying (ex. start buying -1) second index is signal size you will start selling (ex. -1)
close = ignore just set to 0
title= name of backtest, usually i put signal


'''
'''
runBackTest_Binary(test, True, True)
pass test back test object here,
pass True, True
'''
df, data_out, trade_df, fig = runBackTest_Binary(test, True, True)

