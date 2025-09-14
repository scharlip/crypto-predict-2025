from datetime import datetime, timedelta
from typing import List, Tuple

from pandas import DataFrame

from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
from models.MidpointModelPredictor import MidpointPredictorModel
from models.BaseModel import TransctionType


class PerfectMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, ds: SingleStepMidpointCoinDataset, threshold: float, lookback: int, lookahead: int):
        super().__init__(threshold=threshold, lookback=lookback, lookahead=lookahead)
        self.df = ds.df

    def predict_lookahead_window(self, lookback_window: DataFrame) -> List[float]:
        end_timestamp = lookback_window.iloc[-1]["Open time"]
        end_index = self.df[self.df["Open time"] == end_timestamp].index.tolist()[0]
        window_df = self.df[end_index + 1 : end_index + self.lookahead + 1]
        return window_df["Midpoint"].tolist()