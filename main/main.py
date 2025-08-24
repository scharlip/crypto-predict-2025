from datetime import datetime

import torch
from torch import optim

from common.common import CoinType, Exchange
from input.input import MidpointCoinDataset
from models.SimpleLSTMMidpointPredictorModel import SimpleLSTMMidpointPredictorModel
from train.train import train_loop

ds = MidpointCoinDataset(coin_type=CoinType.ETH, exchange=Exchange.Coinbase,
                         date_range_filter=[datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])
model = SimpleLSTMMidpointPredictorModel(0.02, 60)

train_loop(ds=ds, model=model, device=torch.device("cpu"), optimizer=optim.Adam(model.parameters()))