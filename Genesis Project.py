import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


'''Run through and visualize the percent change of BTC and the Z scores
example Z score -2.35 showing that 99% of the data has a percent return greater than -10.7%
while a Z score of 0.459 means that 32% of data had a percent return greater than 1.93%'''
#Statistical Analysis of Percent Change values and Z scores

Scaler = StandardScaler()
data = pd.read_csv("cryptodata.csv")
plotData = (data["btc_price_usd_close"].pct_change() * 100).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="pct_change")
plt.xlim((-5,5))
data["pct_change"] = plotData
signal_data = pd.DataFrame(data["date"], columns = ["date"])
#Gaussian
signal_data["pct_change"] = plotData
signal_data["z_scores_pct_change"] = plotter

#Signal 1 Analysis ETH_BTC_RATIO_CLOSE
data["eth_btc_ratio_close"] = data["eth_price_usd_close"] / data["btc_price_usd_close"]
plotData = (data["eth_btc_ratio_close"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="ETH_BTC_RATIO_CLOSE")
plt.xlim(-5,5)
signal_data["eth_btc_ratio_close"] = plotData
signal_data["z_scores_eth_btc_ratio_close"] = plotter

#Gaussian
#Signal 2 Analysis BTC_FUNDING_RATES
plotData = (data["btc_funding_rates"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Funding_Rates")
plt.xlim(-5,5)
signal_data["btc_funding_rates"] = plotData
signal_data["z_scores_btc_funding_rates"] = plotter

#Signal 3 Analysis BTC_MPI
plotData = (data["btc_mpi"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_MPI")
plt.xlim(-5,5)
signal_data["btc_mpi"] = plotData
signal_data["z_scores_btc_mpi"] = plotter

#Gaussian
#Signal 4 Analysis BTC_ESTIMATED_LEVERAGE_RATIO
plotData = (data["btc_estimated_leverage_ratio"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Estimated_leverage_Ratio")
plt.xlim(-5,5)
signal_data["btc_estimated_leverage_ratio"] = plotData
signal_data["z_scores_btc_estimated_leverage_ratio"] = plotter

#Signal 5 Analysis BTC_STABLECOIN_SUPPLY_RATIO
plotData = (data["btc_stablecoin_supply_ratio"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Stablecoin_Supply_Ratio")
plt.xlim(-5,5)
signal_data["btc_stablecoin_supply_ratio"] = plotData
signal_data["z_scores_btc_stablecoin_supply_ratio"] = plotter

#Signal 6 Analysis BTC_OPEN_INTEREST
plotData = (data["btc_open_interest"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Open_Interest")
plt.xlim(-5,5)
signal_data["btc_open_interest"] = plotData
signal_data["z_scores_btc_open_interest"] = plotter

#Gaussian
#Signal 7 Analysis BTC_NETFLOW_TOTAL
plotData = (data["btc_netflow_total"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Netflow_Total")
plt.xlim(-5,5)
signal_data["btc_netflow_total"] = plotData
signal_data["z_scores_btc_netflow_total"] = plotter

#Signal 8 Analysis BTC_RESERVE
plotData = (data["btc_reserve"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_RESERVE")
plt.xlim(-5,5)
signal_data["btc_reserve"] = plotData
signal_data["z_scores_btc_reserve"] = plotter



#Correlation Matrix
z_with_pct = signal_data[signal_data.columns[::2]]
del z_with_pct["date"]
corr_m_all = z_with_pct.corr()
corr_only_s2p = corr_m_all["z_scores_pct_change"]
corr_only_s2p = corr_only_s2p.iloc[1:]
corr_only_s2p = corr_only_s2p.to_frame()


#Testing with 1 STDEV
z_with_pct_1 = z_with_pct[(z_with_pct>1.0) | (z_with_pct<-1.0)]
corr_only_s2p_1 = z_with_pct_1.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_1 = corr_only_s2p_1.to_frame()
corr_only_s2p_1.columns = ['Z Scores pct_change 1 STDEV']

#Testing with 1.5 STDEV
z_with_pct_1_5 = z_with_pct[(z_with_pct>1.5) | (z_with_pct<-1.5)]
corr_only_s2p_1_5 = z_with_pct_1_5.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_1_5 = corr_only_s2p_1_5.to_frame()
corr_only_s2p_1_5.columns = ['Z Scores pct_change 1.5 STDEV']

#Testing with 2 STDEV
z_with_pct_2 = z_with_pct[(z_with_pct>2.0) | (z_with_pct<-2.0)]
corr_only_s2p_2 = z_with_pct_2.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_2 = corr_only_s2p_2.to_frame()
corr_only_s2p_2.columns = ['Z Scores pct_change 2 STDEV']

#Showing all
final_corr = pd.concat([corr_only_s2p, corr_only_s2p_1, corr_only_s2p_1_5, corr_only_s2p_2], axis = 1)
final_corr = final_corr.sort_values(by = ["Z Scores pct_change 2 STDEV"], ascending=False,)
