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

    def predict_future_window(self, past_window: DataFrame) -> DataFrame:
        end_timestamp = past_window.iloc[-1]["Open time"]
        end_index = self.df[self.df["Open time"] == end_timestamp].index.tolist()[0]

        window_df = self.df[end_index + 1 : end_index + self.lookahead + 1]

        return window_df

    def decision(self,
                   current_time: datetime,
                   future_window: DataFrame,
                   last_purchased_price: float,
                   currently_have_usd: bool) -> Tuple[TransctionType, int]:

        # TODO: cache the window by current_time so that these scans go faster
        min_idx = future_window["Midpoint"].idxmin()
        max_idx = future_window["Midpoint"].idxmax()
        min_value = future_window.loc[min_idx]["Midpoint"].tolist()
        max_value = future_window.loc[max_idx]["Midpoint"].tolist()
        min_datetime = future_window.loc[min_idx]["Open time"].to_pydatetime()
        max_datetime = future_window.loc[max_idx]["Open time"].to_pydatetime()

        if currently_have_usd:
            # buy at the predicted minimum if the maximum is more than the minimum + the threshold
            if max_value > min_value * (1 + self.threshold) and max_idx > min_idx:
                return (TransctionType.Buy, min_datetime)
            else:
                return (TransctionType.Hold, None)
        else:
            # if we predict a value above the threshold, buy there
            if max_value > last_purchased_price * (1 + self.threshold):
                return (TransctionType.Sell, max_datetime)

            # if we predict a value 3x below the threshold, bail out right now
            elif min_value < last_purchased_price * (1 - 3 * self.threshold):
                return (TransctionType.Sell, current_time)
            # otherwise do nothing
            else:
                return (TransctionType.Hold, None)

    def buy_sell_hold_decision(
            self,
            current_time: datetime,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool) -> Tuple[TransctionType, int]:

        future_window = self.predict_future_window(past_window)
        return self.decision(current_time, past_window, future_window, last_purchased_price, currently_have_usd)

    def forward(self, x):
        raise NotImplementedError("This model doesn't get trained.")