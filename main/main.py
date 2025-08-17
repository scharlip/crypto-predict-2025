from backtest.backtest import run_backtest
from common.common import CoinType, Exchange
from input.input import MidpointCoinDataset
from models.NoisySourcePredictorModel import NoisySourceMidpointPredictorModel
from models.PerfectMidpointPredictorModel import PerfectMidpointPredictorModel

ds = MidpointCoinDataset(coin_type=CoinType.ETH, exchange=Exchange.Coinbase)
model = PerfectMidpointPredictorModel(ds, 0.02, 60)

profit = run_backtest(ds, model)