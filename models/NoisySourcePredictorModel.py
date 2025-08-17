from datetime import datetime, timedelta
from typing import List, Tuple

from pandas import DataFrame

from input.input import MidpointCoinDataset
from models.PerfectMidpointPredictorModel import PerfectMidpointPredictorModel
from models.base import TransctionType
import random

class NoisySourceMidpointPredictorModel(PerfectMidpointPredictorModel):

    def __init__(self, ds: MidpointCoinDataset, threshold: float, lookahead: int, artificial_noise_pctg: float = 0.01) -> List[float]:
        super().__init__(ds, threshold, lookahead)
        self.add_noise_function = self.__add_noise_function(artificial_noise_pctg)

    def __add_noise_function(self, artificial_noise_pctg: float) :
        def __noisify(val: float):
            random.seed(val)
            return val * random.uniform(1.0 - artificial_noise_pctg, 1.0 + artificial_noise_pctg)
        return __noisify

    def predict_future_window(self, past_window: DataFrame) -> DataFrame:
        perfect_prediction_window = super().predict_future_window(past_window).copy()
        perfect_prediction_window["Midpoint"] = perfect_prediction_window["Midpoint"].apply(self.add_noise_function)
        return perfect_prediction_window

    def buy_sell_hold_decision(
            self,
            current_time: datetime,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool) -> Tuple[TransctionType, int]:

        future_window = self.predict_future_window(past_window)
        return super().decision(current_time, past_window, future_window, last_purchased_price, currently_have_usd)