from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

from pandas import DataFrame

from models.BaseModel import BaseModel, TransctionType

@dataclass
class CacheEntry:
    current_time: datetime

    min_idx: int
    min_value: float

    max_idx: int
    max_value: float

class MidpointPredictorModel(BaseModel):

    def __init__(self, threshold: float, lookahead: int):
        self.threshold = threshold
        self.lookahead = lookahead
        self.cache = None

    def predict_future_window(self, past_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def decision(self,
                   current_time: datetime,
                   future_window: DataFrame,
                   last_purchased_price: float,
                   currently_have_usd: bool) -> Tuple[TransctionType, int]:

        # caching logic is below. doesn't appear to speed things up much though

        '''
        # try to use cache entry if it's present and was for the previous minute
        if self.cache is not None and current_time - self.cache.current_time == timedelta(minutes = 1):
            cached = self.cache
            trailing_point = future_window.iloc[-1]["Midpoint"].tolist()

            min_value = trailing_point if trailing_point < cached.min_value else  cached.min_value
            min_idx = future_window.index[-1] if trailing_point < cached.min_value else cached.min_idx

            max_value = trailing_point if trailing_point > cached.max_value else  cached.max_value
            max_idx = future_window.index[-1] if trailing_point > cached.max_value else cached.max_idx

            # if the min index has fallen outside the index range, recalculate it
            if min_idx < future_window.index[0]:
                min_idx = future_window["Midpoint"].idxmin()
                min_value = future_window.loc[min_idx]["Midpoint"].tolist()

            # if the max index has fallen outside the index range, recalculate it
            if max_idx < future_window.index[0]:
                max_idx = future_window["Midpoint"].idxmax()
                max_value = future_window.loc[max_idx]["Midpoint"].tolist()

            self.cache = CacheEntry(
                current_time=current_time,
                min_idx=min_idx,
                min_value=min_value,
                max_idx=max_idx,
                max_value=max_value
            )
        # otherwise recalculate
        else:
            min_idx = future_window["Midpoint"].idxmin()
            max_idx = future_window["Midpoint"].idxmax()
            min_value = future_window.loc[min_idx]["Midpoint"].tolist()
            max_value = future_window.loc[max_idx]["Midpoint"].tolist()

            self.cache = CacheEntry(
                current_time=current_time,
                min_idx=min_idx,
                min_value=min_value,
                max_idx=max_idx,
                max_value=max_value
            )
        '''

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

    def forward(self, x):
        raise NotImplementedError("Implemented in subclasses")