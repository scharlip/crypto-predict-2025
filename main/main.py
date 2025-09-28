from datetime import datetime

import torch
from torch import optim

from backtest import backtest
from common.common import CoinType, Exchange
from input.FullWindowMidpointCoinDataset import FullWindowMidpointCoinDataset
from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
from models.midpoint.from_source_data.NoisySourcePredictorModel import NoisySourceMidpointPredictorModel
from models.midpoint.from_source_data.PerfectMidpointPredictorModel import PerfectMidpointPredictorModel
from models.midpoint.lstm.FullWindowLSTMEncoderDecoderMidpointPredictorModel import \
    FullWindowLSTMEncoderDecoderMidpointPredictorModel
from models.midpoint.lstm.FullWindowLSTMMidpointPredictorModel import FullWindowLSTMMidpointPredictorModel
from models.midpoint.lstm.SingleStepLSTMMidpointPredictorModel import SingleStepLSTMMidpointPredictorModel
from train.train import train_loop

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
DATA_DIR = "{}/data".format(BASE_DIR)
MODEL_SAVE_DIR = "{}/models".format(BASE_DIR)
LOG_DIR = "{}/logs".format(BASE_DIR)

lookback = 60
lookahead = 60

torch.set_default_dtype(torch.float64)

single_step_ds =  SingleStepMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase,lookback=lookback, lookahead=lookahead,
                         date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])
full_ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead,
                                   date_range_filter=[datetime(2024, 12, 28, 0, 0), datetime(2025, 1, 1, 0, 0)])

trainable_models = [
    (single_step_ds, SingleStepLSTMMidpointPredictorModel(0.012, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0, normalizer=single_step_ds.normalizer)),
    (full_ds, FullWindowLSTMEncoderDecoderMidpointPredictorModel(threshold=0.012, lookback=lookback, lookahead=lookahead, hidden_size=50, num_layers=1, dropout=0.0, normalizer=full_ds.normalizer)),
    (full_ds, FullWindowLSTMMidpointPredictorModel(0.012, lookback=lookback, lookahead=lookahead, hidden_size=300, num_layers=1, dropout=0.0, normalizer=full_ds.normalizer))
]

non_trainable_models = [
    PerfectMidpointPredictorModel(single_step_ds, 0.012, lookback=lookback, lookahead=lookahead),
    NoisySourceMidpointPredictorModel(single_step_ds, 0.012, lookback=lookback, lookahead=lookahead)
]

for (ds, model) in trainable_models:
    train_loop(ds=ds, model=model, batch_size=64, optimizer=optim.Adam(model.parameters()),
               model_save_dir=MODEL_SAVE_DIR, base_log_dir=LOG_DIR, epochs_per_backtest=1, epochs=2)

for model in non_trainable_models:
    backtest.run_backtest(ds=single_step_ds, model=model, log_file_name="{}/test.log".format(LOG_DIR))

'''
ds = FullWindowMidpointCoinDataset(csv_dir = DATA_DIR, coin_type=CoinType.ETH, exchange=Exchange.Coinbase, lookback=lookback, lookahead=lookahead, use_normalized_data=True,
                                   date_range_filter=[datetime(2023, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)])

saved_path = ("/Users/scharlip/Desktop/crypto-predict/models/FullWindowLSTMMidpointPredictorModel_lookback_60_lookahead_60_hidden_300_layers_1_dropout_0.0_normalized_True/" + \
              "model|epoch_30|batch_size_128|avg_train_loss_0.005357253954487787|avg_validation_loss_0.0034955233605923055.pth")

model = torch.load(saved_path, weights_only=False, map_location=torch.device('cpu'))
model.device = torch.device("cpu")
backtest.run_backtest(ds=ds, model=model, log_file_name="LOG_DIR/{}".format("TODO"))
'''
