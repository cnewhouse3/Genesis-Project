import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mtick
from scipy import stats
import pingouin as pg

'''Run through and visualize the percent change of BTC and the Z scores
example Z score -2.35 showing that 99% of the data has a percent return greater than -10.7%
while a Z score of 0.459 means that 32% of data had a percent return greater than 1.93%'''
#Statistical Analysis of Percent Change values and Z scores

Scaler = StandardScaler()
data = pd.read_csv("cryptodata.csv")
plotData = (data["btc_price_usd_close"].pct_change() * 100).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="pct_change")
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
#sns.displot(plotter).set(Title="ETH_BTC_RATIO_CLOSE")
plt.xlim(-5,5)
signal_data["eth_btc_ratio_close"] = plotData
signal_data["z_scores_eth_btc_ratio_close"] = plotter

#Gaussian
#Signal 2 Analysis BTC_FUNDING_RATES
plotData = (data["btc_funding_rates"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_Funding_Rates")
plt.xlim(-5,5)
signal_data["btc_funding_rates"] = plotData
signal_data["z_scores_btc_funding_rates"] = plotter

#Signal 3 Analysis BTC_MPI
plotData = (data["btc_mpi"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_MPI")
plt.xlim(-5,5)
signal_data["btc_mpi"] = plotData
signal_data["z_scores_btc_mpi"] = plotter

#Gaussian
#Signal 4 Analysis BTC_ESTIMATED_LEVERAGE_RATIO
plotData = (data["btc_estimated_leverage_ratio"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_Estimated_leverage_Ratio")
plt.xlim(-5,5)
signal_data["btc_estimated_leverage_ratio"] = plotData
signal_data["z_scores_btc_estimated_leverage_ratio"] = plotter

#Signal 5 Analysis BTC_STABLECOIN_SUPPLY_RATIO transformed log to get better data
plotData = np.log((data["btc_stablecoin_supply_ratio"]).values)
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_Stablecoin_Supply_Ratio")
plt.xlim(-5,5)
signal_data["btc_stablecoin_supply_ratio"] = plotData
signal_data["z_scores_btc_stablecoin_supply_ratio"] = plotter

#Signal 6 Analysis BTC_OPEN_INTEREST
plotData = (data["btc_open_interest"]).values ** (1/3)
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_Open_Interest")
plt.xlim(-5,5)
signal_data["btc_open_interest"] = plotData
signal_data["z_scores_btc_open_interest"] = plotter

#Gaussian
#Signal 7 Analysis BTC_NETFLOW_TOTAL
plotData = (data["btc_netflow_total"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_Netflow_Total")
plt.xlim(-5,5)
signal_data["btc_netflow_total"] = plotData
signal_data["z_scores_btc_netflow_total"] = plotter

#Signal 8 Analysis BTC_RESERVE
plotData = (data["btc_reserve"]).values 
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
#sns.displot(plotter).set(Title="BTC_RESERVE")
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

allstats = z_with_pct.pairwise_corr(method='spearman',padjust='holm')
pval = allstats[['X','Y','n','r','p-unc']].round(3)
stats = pval[:8]
stats['r2'] = stats['r'] ** 2
stats.round(3)



signal_data = signal_data.replace(np.nan, 0)

#Testing with 1 STDEV
z_with_pct_1 = z_with_pct[(z_with_pct>1.0) | (z_with_pct<-1.0)]
corr_only_s2p_1 = z_with_pct_1.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_1 = corr_only_s2p_1.to_frame()
corr_only_s2p_1.columns = ['Z Scores pct_change 1 STDEV']

allstats1 = z_with_pct_1.pairwise_corr(method='spearman',padjust='holm')
pval = allstats1[['X','Y','n','r','p-unc']].round(3)
stats_1 = pval[:8]
stats_1['r2'] = stats_1['r'] ** 2
stats_1.round(3)

#Testing with 1.5 STDEV
z_with_pct_1_5 = z_with_pct[(z_with_pct>1.5) | (z_with_pct<-1.5)]
corr_only_s2p_1_5 = z_with_pct_1_5.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_1_5 = corr_only_s2p_1_5.to_frame()
corr_only_s2p_1_5.columns = ['Z Scores pct_change 1.5 STDEV']

allstats1_5 = z_with_pct_1_5.pairwise_corr(method='spearman',padjust='holm')
pval = allstats1_5[['X','Y','n','r','p-unc']].round(3)
stats_1_5 = pval[:8]
stats_1_5['r2'] = stats_1_5['r'] ** 2
stats_1_5.round(3)

#Testing with 2 STDEV
z_with_pct_2 = z_with_pct[(z_with_pct>2.0) | (z_with_pct<-2.0)]
corr_only_s2p_2 = z_with_pct_2.corr()["z_scores_pct_change"].iloc[1:]
corr_only_s2p_2 = corr_only_s2p_2.to_frame()
corr_only_s2p_2.columns = ['Z Scores pct_change 2 STDEV']

allstats2 = z_with_pct_2.pairwise_corr(method='spearman',padjust='holm')
pval = allstats2[['X','Y','n','r','p-unc']].round(3)
stats_2 = pval[:8]
stats_2['r2'] = stats_2['r'] ** 2
stats_2.round(3)

#Creating Correlation Matrix of all Signals to Price


#Showing all
final_corr = pd.concat([corr_only_s2p, corr_only_s2p_1, corr_only_s2p_1_5, corr_only_s2p_2], axis = 1)
final_corr = final_corr.sort_values(by = ["Z Scores pct_change 2 STDEV"], ascending=False,)


backtest = z_with_pct.drop(["z_scores_pct_change"], axis = 1)
backtest["date"] = data["date"]
backtest["btc_price_usd_close"] = data["btc_price_usd_close"]
backtest = backtest.replace(np.nan, 0)
backtest.to_csv(r'C:\Users\ctnsc\backtest_dataframe.csv')

final_r2 = final_corr ** 2


fig, ax = plt.subplots(2,2)
ax[0, 0].scatter(z_with_pct["z_scores_btc_reserve"].values, signal_data["pct_change"].values)
ax[0, 1].scatter(z_with_pct_1["z_scores_btc_reserve"].values, signal_data["pct_change"].values)
ax[1,0].scatter(z_with_pct_1_5["z_scores_btc_reserve"].values, signal_data["pct_change"].values)
ax[1,1].scatter(z_with_pct_2["z_scores_btc_reserve"].values, signal_data["pct_change"].values)


ax[0, 0].set_title('All Z_scores + Price Change')
ax[0, 0].set_xlabel('z_scores_btc_reserve')
ax[0, 0].set_ylabel('PCT Change in BTC')

ax[0, 1].set_title('Z Scores + Price 1 STDEV')
ax[0, 1].set_xlabel('z_scores_btc_reserve')
ax[0, 1].set_ylabel('PCT Change in BTC')

ax[1,0].set_title('Z Scores + Price 1.5 STDEV')
ax[1,0].set_xlabel('z_scores_btc_reserve')
ax[1,0].set_ylabel('PCT Change in BTC')

ax[1,1].set_title('Z Scores + Price 2 STDEV')
ax[1,1].set_xlabel('z_scores_btc_reserve')
ax[1,1].set_ylabel('PCT Change in BTC')

ax[0,0].set_xlim([-3,3])
ax[0,1].set_xlim([-3,3])
ax[1,0].set_xlim([-3,3])
ax[1,1].set_xlim([-3,3])

ax[0,0].yaxis.set_major_formatter(mtick.PercentFormatter())
ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter())
ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter())
ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter())

ax[0,0].set_yticks(np.arange(-20, 70, 20))
ax[0,1].set_yticks(np.arange(-20, 30, 10))
ax[1,0].set_yticks(np.arange(-20, 30, 10))
ax[1,1].set_yticks(np.arange(-20, 30, 10))
fig.tight_layout()



plt.show()
