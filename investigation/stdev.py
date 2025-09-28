from datetime import datetime

from common.common import CoinType, Exchange

from input.FullWindowMidpointCoinDataset import FullWindowMidpointCoinDataset

import matplotlib.pyplot as plt

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
DATA_DIR = "{}/data".format(BASE_DIR)
coin_type = CoinType.ADA
exchange = Exchange.Coinbase

lookback = 60
lookahead = 60

ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=coin_type, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                                   date_range_filter=[datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])

rolling_stddev = ds.df["Midpoint"].rolling(60).std()
scaled_rolling_stddev = rolling_stddev / ds.df["Midpoint"]
above_threshold = (scaled_rolling_stddev > 0.012).astype(int)

scaled_rolling_stddev.plot()
plt.show()