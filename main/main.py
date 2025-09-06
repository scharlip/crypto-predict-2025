from datetime import datetime

from torch import optim

from common.common import CoinType, Exchange
from input.FullWindowMidpointCoinDataset import FullWindowMidpointCoinDataset
from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
from models.FullWindowLSTMMidpointPredictorModel import FullWindowLSTMMidpointPredictorModel
from models.SingleStepLSTMMidpointPredictorModel import SingleStepLSTMMidpointPredictorModel
from train.train import train_loop

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
DATA_DIR = "{}/data".format(BASE_DIR)

'''
ds = SingleStepMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase,
                         date_range_filter=[datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = SingleStepLSTMMidpointPredictorModel(0.012, 60, hidden_size=300, num_layers=3, dropout=0.1)
'''

ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase,
                         date_range_filter=[datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = FullWindowLSTMMidpointPredictorModel(0.012, 60, hidden_size=300, num_layers=3, dropout=0.1)

train_loop(ds=ds, model=model, optimizer=optim.Adam(model.parameters()))