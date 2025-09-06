from datetime import datetime, timedelta
from typing import Tuple, List

from pandas import DataFrame

from models.BaseModel import BaseModel, TransctionType

class MidpointPredictorModel(BaseModel):

    def __init__(self, threshold: float, lookahead: int):
        super().__init__()
        self.threshold = threshold
        self.lookahead = lookahead
        self.cache = None

    def predict_future_window(self, past_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def decision(self,
                   current_time: datetime,
                   future_window: List[float],
                   last_purchased_price: float,
                   currently_have_usd: bool) -> Tuple[TransctionType, int]:

        min_idx = None
        max_idx = None
        min_value = None
        max_value = None

        for (idx, val) in enumerate(future_window):
            if min_value is None or min_value > val:
                min_value = val
                min_idx = idx

            if max_value is None or max_value < val:
                max_value = val
                max_idx = idx

        min_datetime = current_time + timedelta(minutes=min_idx)
        max_datetime = current_time + timedelta(minutes=max_idx)

        if currently_have_usd:
            # buy at the predicted minimum if the maximum is more than the minimum + the threshold
            if max_value > min_value * (1 + self.threshold) and max_idx > min_idx:
                return (TransctionType.Buy, min_datetime)
            else:
                return (TransctionType.Hold, None)
        else:
            # if we predict a value above the threshold, sell there
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