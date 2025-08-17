from datetime import datetime, timedelta
from typing import List, Tuple

from pandas import DataFrame

from input.input import MidpointCoinDataset
from models.base import BaseModel, TransctionType


class PerfectMidpointPredictorModel(BaseModel):

    def __init__(self, ds: MidpointCoinDataset, threshold: float, lookahead: int):
        self.df = ds.df
        self.threshold = threshold
        self.lookahead = lookahead

    def __predict_future_window(self, past_window: DataFrame) -> List[float]:
        '''
        with_noise = []
        for v in perfect_prediction:
            random.seed(v)
            with_noise.append(v * random.uniform(1.0 - artificial_noise_pctg, 1.0 + artificial_noise_pctg))
        '''


        end_timestamp = past_window.iloc[-1]["Open time"]
        end_index = self.df[self.df["Open time"] == end_timestamp].index.tolist()[0]

        window_df = self.df[end_index + 1 : end_index + self.lookahead + 1]

        return window_df["Midpoint"].tolist()

    def buy_sell_hold_decision(
            self,
            current_time: datetime,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool) -> Tuple[TransctionType, int]:

        future_window = self.__predict_future_window(past_window)

        min_value = float('inf')
        min_index = -1
        max_value = float('-inf')
        max_index = -1

        for (idx, val) in enumerate(future_window):
            if val > max_value:
                max_value = val
                max_index = idx

            if val < min_value:
                min_value = val
                min_index = idx

        if currently_have_usd:
            # buy at the predicted minimum if the maximum is more than the minimum + the threshold
            if max_value > min_value * (1 + self.threshold) and max_index > min_index:
                return (TransctionType.Buy, current_time + timedelta(minutes = min_index))
            else:
                return (TransctionType.Hold, None)
        else:
            # if we predict a value above the threshold, buy there
            if max_value > last_purchased_price * (1 + self.threshold):
                return (TransctionType.Sell, current_time + timedelta(minutes = max_index))

            # if we predict a value 3x below the threshold, bail out right now
            elif min_value < last_purchased_price * (1 - 3 * self.threshold):
                return (TransctionType.Sell, current_time)
            # otherwise do nothing
            else:
                return (TransctionType.Hold, None)

    def forward(self, x):
        raise NotImplementedError("This model doesn't get trained.")