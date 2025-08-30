from datetime import datetime

import torch
from torch import optim

from common.common import CoinType, Exchange
from input.input import MidpointCoinDataset
from models.SimpleLSTMMidpointPredictorModel import SimpleLSTMMidpointPredictorModel
from train.train import train_loop

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
DATA_DIR = "{}/data".format(BASE_DIR)

ds = MidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase,
                         date_range_filter=[datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = SimpleLSTMMidpointPredictorModel(0.012, 60, hidden_size=300, num_layers=3, dropout=0.1)

train_loop(ds=ds, model=model, device=torch.device("cpu"), optimizer=optim.Adam(model.parameters()))