import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy import stats

Scaler = StandardScaler()
data = pd.read_csv("cryptodata.csv")
plotData = (data["btc_price_usd_close"].pct_change() * 100).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
'''
sns.displot(plotter).set(Title="pct_change")
plt.xlim((-5,5))
data["pct_change"] = plotData
'''
signal_data = pd.DataFrame(data["date"], columns = ["date"])
#Gaussian
signal_data["pct_change"] = plotData
signal_data["z_scores_pct_change"] = plotter



#Signal 5 Analysis BTC_STABLECOIN_SUPPLY_RATIO
plotData = (data["btc_stablecoin_supply_ratio"]).values
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Stablecoin_Supply_Ratio")
plt.xlim(-5,5)
signal_data["btc_stablecoin_supply_ratio"] = plotData
signal_data["z_scores_btc_stablecoin_supply_ratio"] = plotter

#Signal 5 BoxCox StableCoin_supply ratio
plotData, l = stats.boxcox((data["btc_stablecoin_supply_ratio"]).values)
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Stablecoin_Supply_Ratio BoxCox")
plt.xlim(-5,5)
signal_data["btc_stablecoin_supply_ratio"] = plotData
signal_data["z_scores_btc_stablecoin_supply_ratio"] = plotter

#Signal 5 log Stablecoin
plotData = np.log((data["btc_stablecoin_supply_ratio"]).values)
plotter = Scaler.fit_transform(plotData.reshape(-1,1))
sns.displot(plotter).set(Title="BTC_Stablecoin_Supply_Ratio Log")
plt.xlim(-5,5)
signal_data["btc_stablecoin_supply_ratio"] = plotData
signal_data["z_scores_btc_stablecoin_supply_ratio"] = plotter

#Signal 5 sqrt