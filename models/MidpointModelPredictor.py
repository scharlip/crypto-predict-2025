from pandas import DataFrame

from models.BaseModel import BaseModel, TransctionType

class MidpointPredictorModel(BaseModel):

    def __init__(self, threshold: float, lookback: int, lookahead: int):
        super().__init__()
        self.threshold = threshold
        self.lookback = lookback
        self.lookahead = lookahead
        self.cache = None

    def predict_lookahead_window(self, lookback_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def forward(self, x):
        raise NotImplementedError("Implemented in subclasses")