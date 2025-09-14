from datetime import datetime

from torch import optim
import numpy as np
import torch

from common.common import CoinType, Exchange
from input.FullWindowMidpointCoinDataset import FullWindowMidpointCoinDataset
from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
from models.FullWindowLSTMEncoderDecoderMidpointPredictorModel import FullWindowLSTMEncoderDecoderMidpointPredictorModel
from models.FullWindowLSTMMidpointPredictorModel import FullWindowLSTMMidpointPredictorModel
from models.NoisySourcePredictorModel import NoisySourceMidpointPredictorModel
from models.PerfectMidpointPredictorModel import PerfectMidpointPredictorModel
from models.SingleStepLSTMMidpointPredictorModel import SingleStepLSTMMidpointPredictorModel
from train.train import train_loop

from backtest import backtest

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
DATA_DIR = "{}/data".format(BASE_DIR)

lookback = 60
lookahead = 15


ds = SingleStepMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                         date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = PerfectMidpointPredictorModel(ds, 0.012, lookback=lookback, lookahead=lookahead)


'''
ds = SingleStepMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                         date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = NoisySourceMidpointPredictorModel(ds, 0.012, lookback=lookback, lookahead=lookahead)
'''

'''
ds = SingleStepMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase,lookback=lookback, lookahead=lookahead,
                         date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = SingleStepLSTMMidpointPredictorModel(0.012, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0)
'''

'''
ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                                   date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = FullWindowLSTMMidpointPredictorModel(0.012, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0)
'''

'''
ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                                   date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = FullWindowLSTMEncoderDecoderMidpointPredictorModel(threshold=0.012, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0)
'''

'''
ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                                   date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = FullWindowLSTMMidpointPredictorModel(threshold=0.2, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0)
'''
train_loop(ds=ds, model=model, optimizer=optim.Adam(model.parameters()))
