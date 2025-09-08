from datetime import datetime, timedelta
from typing import List, Tuple

from pandas import DataFrame

from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
from models.BaseModel import TransctionType
import random

from models.MidpointModelPredictor import MidpointPredictorModel


class NoisySourceMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, ds: SingleStepMidpointCoinDataset, threshold: float, lookahead: int, artificial_noise_pctg: float = 0.01) -> List[float]:
        super().__init__(threshold, lookahead)
        self.df = ds.df
        self.add_noise_function = self.__add_noise_function(artificial_noise_pctg)

    def __add_noise_function(self, artificial_noise_pctg: float) :
        def __noisify(val: float):
            random.seed(val)
            return val * random.uniform(1.0 - artificial_noise_pctg, 1.0 + artificial_noise_pctg)
        return __noisify

    def predict_future_window(self, past_window: DataFrame) -> List[float]:
        end_timestamp = past_window.iloc[-1]["Open time"]
        end_index = self.df[self.df["Open time"] == end_timestamp].index.tolist()[0]
        perfect_prediction_window = self.df[end_index + 1: end_index + self.lookahead + 1].copy()
        perfect_prediction_window["Midpoint"] = perfect_prediction_window["Midpoint"].apply(self.add_noise_function)
        return perfect_prediction_window["Midpoint"].tolist()